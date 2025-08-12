"""
RBBD (Representation-Based Backdoor Defense) Implementation
Main defense system implementing the complete RBBD framework from the paper
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import copy


class RBBDDefense:
    """
    Representation-Based Backdoor Defense (RBBD) system
    Implements the complete defense framework with adaptive thresholding,
    distribution-aware filtering, and risk-based client classification
    """
    
    def __init__(self, config):
        """
        Initialize RBBD defense system
        
        Args:
            config: Configuration object with defense parameters
        """
        self.config = config
        
        # Client evidence tracking
        self.client_evidence = defaultdict(lambda: {
            'suspicion': 0.0,
            'consecutive_bad': 0,
            'is_quarantined': False,
            'quarantine_end_round': 0,
            'risk_history': deque(maxlen=10),
            'last_risk_score': 0.0
        })
        
        # Adaptive defense parameters
        self.defense_intensity = getattr(config, 'INITIAL_DEFENSE_INTENSITY', 0.4)
        self.intensity_ramp_rate = getattr(config, 'INTENSITY_RAMP_UP_RATE', 0.015)
        self.intensity_cool_rate = getattr(config, 'INTENSITY_COOL_DOWN_RATE', 0.01)
        
        # Thresholds
        self.evidence_decay = getattr(config, 'EVIDENCE_DECAY', 0.6)
        self.tail_bias_threshold = getattr(config, 'TAIL_BIAS_THRESHOLD', 0.03)
        self.rep_shift_threshold = getattr(config, 'REP_SHIFT_THRESHOLD', 1.8)
        self.consecutive_strikes = getattr(config, 'CONSECUTIVE_ATTACK_STRIKES', 3)
        self.quarantine_rounds = getattr(config, 'QUARANTINE_ROUNDS', 1)
        
        # Risk classification thresholds
        self.tau_low = getattr(config, 'TAU_LOW', 0.3)
        self.tau_medium = getattr(config, 'TAU_MEDIUM', 0.6)
        self.tau_high = getattr(config, 'TAU_HIGH', 0.8)
        
        # Performance tracking
        self.performance_history = deque(maxlen=10)
        self.recent_analyses = deque(maxlen=50)
        
    def update_defense_intensity(self, current_asr: float, convergence_status: float = 0.5):
        """
        Update defense intensity based on current attack success rate and convergence
        Implements adaptive thresholding mechanism from the paper
        
        Args:
            current_asr: Current attack success rate (0-100)
            convergence_status: Training convergence status (0-1)
        """
        # Proactively ramp up intensity if attack is succeeding
        if current_asr > 10.0:  # If ASR is above 10%
            ramp_factor = min(2.0, current_asr / 10.0)  # Scale with severity
            self.defense_intensity = min(1.0, 
                self.defense_intensity + self.intensity_ramp_rate * ramp_factor)
        
        # Cool down if system appears clean and converged
        elif current_asr < 2.0 and convergence_status > 0.7:
            self.defense_intensity = max(0.1, 
                self.defense_intensity - self.intensity_cool_rate)
        
        # Moderate adjustment for intermediate cases
        elif current_asr < 5.0:
            self.defense_intensity = max(0.2, 
                self.defense_intensity - self.intensity_cool_rate * 0.5)
        
        # Record performance
        self.performance_history.append({
            'asr': current_asr,
            'intensity': self.defense_intensity,
            'convergence': convergence_status
        })
    
    def analyze_client_suspicion(self, client_analyses: Dict[int, Dict[str, float]], 
                               round_num: int) -> Dict[int, str]:
        """
        Analyze client suspicion levels and update evidence
        Implements the evidence accumulation and suspicion scoring
        
        Args:
            client_analyses: Dictionary mapping client IDs to analysis results
            round_num: Current training round
            
        Returns:
            Dictionary mapping client IDs to risk categories
        """
        client_categories = {}
        
        # Calculate baseline for adaptive thresholds
        if client_analyses:
            all_rep_shifts = [a['rep_shift'] for a in client_analyses.values()]
            baseline_rep_shift = np.percentile(all_rep_shifts, 25) if all_rep_shifts else 0
        else:
            baseline_rep_shift = 0
        
        for client_id, analysis in client_analyses.items():
            evidence = self.client_evidence[client_id]
            
            # Skip updates for quarantined clients
            if evidence['is_quarantined'] and round_num < evidence['quarantine_end_round']:
                client_categories[client_id] = 'quarantined'
                continue
            elif evidence['is_quarantined'] and round_num >= evidence['quarantine_end_round']:
                # Release from quarantine
                evidence['is_quarantined'] = False
                evidence['consecutive_bad'] = 0
                evidence['suspicion'] *= 0.5  # Reduce suspicion upon release
            
            # Determine if client behavior is suspicious
            is_suspicious = self._evaluate_client_behavior(analysis, baseline_rep_shift)
            
            # Update consecutive bad behavior count
            if is_suspicious:
                evidence['consecutive_bad'] += 1
            else:
                evidence['consecutive_bad'] = 0
            
            # Three-strikes rule: Quarantine persistent offenders
            if evidence['consecutive_bad'] >= self.consecutive_strikes:
                evidence['is_quarantined'] = True
                evidence['quarantine_end_round'] = round_num + self.quarantine_rounds
                evidence['suspicion'] = 1.0  # Max out suspicion
                evidence['consecutive_bad'] = 0  # Reset counter
                client_categories[client_id] = 'quarantined'
                continue
            
            # Update suspicion score using exponential moving average
            new_suspicion = 1.0 if is_suspicious else 0.0
            evidence['suspicion'] = (evidence['suspicion'] * self.evidence_decay + 
                                   new_suspicion * (1 - self.evidence_decay))
            
            # Store risk history
            evidence['risk_history'].append(analysis['risk_score'])
            evidence['last_risk_score'] = analysis['risk_score']
            
            # Categorize client based on risk level
            client_categories[client_id] = self._categorize_client_risk(evidence, analysis)
            
            # Store analysis for adaptive threshold updates
            self.recent_analyses.append(analysis)
        
        return client_categories
    
    def _evaluate_client_behavior(self, analysis: Dict[str, float], 
                                baseline_rep_shift: float) -> bool:
        """
        Evaluate if client behavior is suspicious based on analysis results
        
        Args:
            analysis: Client analysis results
            baseline_rep_shift: Baseline representation shift for comparison
            
        Returns:
            True if behavior is suspicious
        """
        # Check representation shift anomaly
        rep_shift_anomaly = analysis['rep_shift'] > baseline_rep_shift + self.rep_shift_threshold
        
        # Check tail bias anomaly
        tail_bias_anomaly = abs(analysis['diff_impact']) > self.tail_bias_threshold
        
        # Check high risk score
        high_risk = analysis['risk_score'] > self.tau_medium
        
        # Client is suspicious if any major anomaly is detected
        return rep_shift_anomaly or tail_bias_anomaly or high_risk
    
    def _categorize_client_risk(self, evidence: Dict, analysis: Dict[str, float]) -> str:
        """
        Categorize client into risk levels based on evidence and current analysis
        
        Args:
            evidence: Client evidence dictionary
            analysis: Current analysis results
            
        Returns:
            Risk category string
        """
        risk_score = analysis['risk_score']
        suspicion = evidence['suspicion']
        
        # Combine current risk with historical suspicion
        combined_risk = 0.7 * risk_score + 0.3 * suspicion
        
        if combined_risk >= self.tau_high:
            return 'high_risk'
        elif combined_risk >= self.tau_medium:
            return 'medium_risk'
        elif combined_risk >= self.tau_low:
            return 'low_risk'
        else:
            return 'trusted'
    
    def calculate_client_weights(self, client_ids: List[int], 
                               client_categories: Dict[int, str],
                               round_num: int) -> Dict[int, float]:
        """
        Calculate aggregation weights for clients based on risk assessment
        Implements risk-based client classification from the paper
        
        Args:
            client_ids: List of client IDs
            client_categories: Risk categories for each client
            round_num: Current training round
            
        Returns:
            Dictionary mapping client IDs to aggregation weights
        """
        weights = {}
        
        for client_id in client_ids:
            evidence = self.client_evidence[client_id]
            category = client_categories.get(client_id, 'trusted')
            
            # Handle quarantined clients
            if (evidence['is_quarantined'] and 
                round_num < evidence['quarantine_end_round']):
                weights[client_id] = 0.0
                continue
            
            # Calculate base weight based on category and defense intensity
            base_weight = self._get_base_weight_for_category(category)
            
            # Apply defense intensity scaling
            if category != 'trusted':
                # More aggressive weight reduction when defense intensity is high
                reduction_factor = evidence['suspicion'] * self.defense_intensity
                final_weight = base_weight * (1 - reduction_factor)
            else:
                final_weight = base_weight
            
            # Ensure weight is non-negative
            weights[client_id] = max(0.0, final_weight)
        
        return weights
    
    def _get_base_weight_for_category(self, category: str) -> float:
        """
        Get base aggregation weight for risk category
        
        Args:
            category: Risk category
            
        Returns:
            Base weight value
        """
        weight_map = {
            'trusted': 1.0,      # Full participation weight
            'low_risk': 0.8,     # Slightly reduced weight
            'medium_risk': 0.5,  # Moderately reduced weight  
            'high_risk': 0.2,    # Heavily reduced weight
            'quarantined': 0.0   # No participation
        }
        
        return weight_map.get(category, 0.5)
    
    def aggregate_updates(self, client_deltas: Dict[int, Dict[str, torch.Tensor]],
                        client_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using calculated weights
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated parameter update
        """
        if not client_deltas:
            return {}
        
        # Calculate total weight for normalization
        total_weight = sum(client_weights.values())
        
        if total_weight == 0:
            # If all clients are excluded, return zero update
            sample_delta = next(iter(client_deltas.values()))
            zero_update = {}
            for key, param in sample_delta.items():
                zero_update[key] = torch.zeros_like(param)
            return zero_update
        
        # Initialize aggregated update
        aggregated_delta = {}
        sample_delta = next(iter(client_deltas.values()))
        
        for param_name in sample_delta.keys():
            aggregated_delta[param_name] = torch.zeros_like(sample_delta[param_name])
        
        # Weighted aggregation
        for client_id, delta in client_deltas.items():
            weight = client_weights.get(client_id, 0.0)
            normalized_weight = weight / total_weight
            
            for param_name, param_delta in delta.items():
                # Convert normalized_weight to the same dtype as param_delta
                weight_tensor = torch.tensor(normalized_weight, dtype=param_delta.dtype, device=param_delta.device)
                aggregated_delta[param_name] += param_delta * weight_tensor
        
        return aggregated_delta
    
    def get_defense_statistics(self) -> Dict[str, float]:
        """
        Get current defense system statistics
        
        Returns:
            Dictionary of defense statistics
        """
        total_clients = len(self.client_evidence)
        quarantined = sum(1 for e in self.client_evidence.values() if e['is_quarantined'])
        
        if self.client_evidence:
            avg_suspicion = np.mean([e['suspicion'] for e in self.client_evidence.values()])
            max_suspicion = max([e['suspicion'] for e in self.client_evidence.values()])
        else:
            avg_suspicion = 0.0
            max_suspicion = 0.0
        
        # Recent performance statistics
        if self.performance_history:
            recent_asr = [p['asr'] for p in self.performance_history]
            avg_recent_asr = np.mean(recent_asr)
        else:
            avg_recent_asr = 0.0
        
        return {
            'defense_intensity': self.defense_intensity,
            'total_clients': total_clients,
            'quarantined_clients': quarantined,
            'avg_suspicion': avg_suspicion,
            'max_suspicion': max_suspicion,
            'avg_recent_asr': avg_recent_asr,
            'quarantine_rate': quarantined / total_clients if total_clients > 0 else 0.0
        }
    
    def reset_client_evidence(self, client_id: int):
        """
        Reset evidence for a specific client (useful for testing)
        
        Args:
            client_id: Client ID to reset
        """
        if client_id in self.client_evidence:
            del self.client_evidence[client_id]
    
    def get_client_risk_summary(self) -> Dict[str, int]:
        """
        Get summary of client risk distribution
        
        Returns:
            Dictionary with counts for each risk category
        """
        categories = defaultdict(int)
        
        for evidence in self.client_evidence.values():
            if evidence['is_quarantined']:
                categories['quarantined'] += 1
            else:
                # Categorize based on current suspicion level
                suspicion = evidence['suspicion']
                if suspicion >= self.tau_high:
                    categories['high_risk'] += 1
                elif suspicion >= self.tau_medium:
                    categories['medium_risk'] += 1
                elif suspicion >= self.tau_low:
                    categories['low_risk'] += 1
                else:
                    categories['trusted'] += 1
        
        return dict(categories)
    
    def should_update_thresholds(self, round_num: int) -> bool:
        """
        Determine if adaptive thresholds should be updated
        
        Args:
            round_num: Current training round
            
        Returns:
            True if thresholds should be updated
        """
        # Update thresholds every 10 rounds or when defense intensity changes significantly
        if round_num % 10 == 0:
            return True
        
        if (self.performance_history and len(self.performance_history) > 1):
            intensity_change = abs(self.performance_history[-1]['intensity'] - 
                                 self.performance_history[-2]['intensity'])
            if intensity_change > 0.1:
                return True
        
        return False