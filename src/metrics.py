import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F


def signal_to_distortion_ratio(estimated: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimated: Estimated signal (batch, channels, samples) or (channels, samples)
        reference: Reference signal (same shape as estimated)
        eps: Small epsilon for numerical stability
        
    Returns:
        sdr: SDR in dB
    """
    # Ensure same shape
    assert estimated.shape == reference.shape
    
    # Flatten to (batch, samples) if needed
    original_shape = estimated.shape
    if len(estimated.shape) == 3:
        batch_size, channels, samples = estimated.shape
        estimated = estimated.reshape(batch_size, -1)
        reference = reference.reshape(batch_size, -1)
    elif len(estimated.shape) == 2:
        estimated = estimated.reshape(1, -1)
        reference = reference.reshape(1, -1)
    
    # Compute SDR
    reference_power = torch.sum(reference ** 2, dim=-1)
    error = estimated - reference
    error_power = torch.sum(error ** 2, dim=-1)
    
    sdr = 10 * torch.log10(reference_power / (error_power + eps) + eps)
    
    return sdr


def signal_to_interference_ratio(estimated: torch.Tensor, 
                                 reference: torch.Tensor,
                                 other_sources: List[torch.Tensor],
                                 eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Signal-to-Interference Ratio (SIR).
    
    Args:
        estimated: Estimated signal (batch, channels, samples)
        reference: Reference signal (same shape as estimated)
        other_sources: List of other source signals that should be suppressed
        eps: Small epsilon for numerical stability
        
    Returns:
        sir: SIR in dB
    """
    # Project estimated onto reference to get the target component
    reference_flat = reference.reshape(reference.shape[0], -1)
    estimated_flat = estimated.reshape(estimated.shape[0], -1)
    
    # Compute projection coefficient
    alpha = torch.sum(estimated_flat * reference_flat, dim=-1, keepdim=True) / \
            (torch.sum(reference_flat ** 2, dim=-1, keepdim=True) + eps)
    
    # Target component
    s_target = alpha * reference_flat
    
    # Interference component (from other sources)
    s_interf = torch.zeros_like(estimated_flat)
    for other_source in other_sources:
        other_flat = other_source.reshape(other_source.shape[0], -1)
        beta = torch.sum(estimated_flat * other_flat, dim=-1, keepdim=True) / \
               (torch.sum(other_flat ** 2, dim=-1, keepdim=True) + eps)
        s_interf += beta * other_flat
    
    # Compute SIR
    target_power = torch.sum(s_target ** 2, dim=-1)
    interf_power = torch.sum(s_interf ** 2, dim=-1)
    
    sir = 10 * torch.log10(target_power / (interf_power + eps) + eps)
    
    return sir


def signal_to_artifacts_ratio(estimated: torch.Tensor,
                              reference: torch.Tensor,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Signal-to-Artifacts Ratio (SAR).
    
    Args:
        estimated: Estimated signal (batch, channels, samples)
        reference: Reference signal (same shape as estimated)
        eps: Small epsilon for numerical stability
        
    Returns:
        sar: SAR in dB
    """
    # Flatten
    reference_flat = reference.reshape(reference.shape[0], -1)
    estimated_flat = estimated.reshape(estimated.shape[0], -1)
    
    # Project estimated onto reference
    alpha = torch.sum(estimated_flat * reference_flat, dim=-1, keepdim=True) / \
            (torch.sum(reference_flat ** 2, dim=-1, keepdim=True) + eps)
    
    # Target component and artifacts
    s_target = alpha * reference_flat
    e_artif = estimated_flat - s_target
    
    # Compute SAR
    target_power = torch.sum(s_target ** 2, dim=-1)
    artif_power = torch.sum(e_artif ** 2, dim=-1)
    
    sar = 10 * torch.log10(target_power / (artif_power + eps) + eps)
    
    return sar


def bss_eval_sources(estimated: torch.Tensor,
                     reference: torch.Tensor,
                     other_sources: List[torch.Tensor] = None,
                     eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    Compute BSS eval metrics (SDR, SIR, SAR).
    
    Args:
        estimated: Estimated signal (batch, channels, samples)
        reference: Reference signal (same shape as estimated)
        other_sources: List of other source signals for SIR computation
        eps: Small epsilon for numerical stability
        
    Returns:
        metrics: Dictionary with 'sdr', 'sir', 'sar' keys
    """
    metrics = {}
    
    # SDR
    metrics['sdr'] = signal_to_distortion_ratio(estimated, reference, eps)
    
    # SIR (if other sources provided)
    if other_sources is not None and len(other_sources) > 0:
        metrics['sir'] = signal_to_interference_ratio(estimated, reference, other_sources, eps)
    
    # SAR
    metrics['sar'] = signal_to_artifacts_ratio(estimated, reference, eps)
    
    return metrics


def compute_separation_metrics(estimated_stems: Dict[str, torch.Tensor],
                               reference_stems: Dict[str, torch.Tensor],
                               stem_names: List[str],
                               eps: float = 1e-8) -> Dict[str, Dict[str, float]]:
    """
    Compute separation metrics for all stems.
    
    Args:
        estimated_stems: Dictionary of estimated stems {stem_name: tensor}
        reference_stems: Dictionary of reference stems {stem_name: tensor}
        stem_names: List of stem names
        eps: Small epsilon for numerical stability
        
    Returns:
        metrics: Nested dictionary {stem_name: {metric_name: value}}
    """
    all_metrics = {}
    
    for stem_name in stem_names:
        estimated = estimated_stems[stem_name]
        reference = reference_stems[stem_name]
        
        # Get other sources for SIR computation
        other_sources = [
            reference_stems[other_stem] 
            for other_stem in stem_names 
            if other_stem != stem_name
        ]
        
        # Compute metrics
        stem_metrics = bss_eval_sources(estimated, reference, other_sources, eps)
        
        # Convert to float and store
        all_metrics[stem_name] = {
            key: value.mean().item() if isinstance(value, torch.Tensor) else value
            for key, value in stem_metrics.items()
        }
    
    # Compute average metrics across all stems
    avg_metrics = {}
    for metric_name in ['sdr', 'sir', 'sar']:
        values = [
            all_metrics[stem][metric_name] 
            for stem in stem_names 
            if metric_name in all_metrics[stem]
        ]
        if values:
            avg_metrics[metric_name] = np.mean(values)
    
    all_metrics['average'] = avg_metrics
    
    return all_metrics


def spectral_convergence(estimated_mag: torch.Tensor, 
                         reference_mag: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral convergence metric.
    
    Args:
        estimated_mag: Estimated magnitude spectrogram
        reference_mag: Reference magnitude spectrogram
        
    Returns:
        sc: Spectral convergence value (lower is better)
    """
    numerator = torch.norm(reference_mag - estimated_mag, p='fro')
    denominator = torch.norm(reference_mag, p='fro')
    
    return (numerator / denominator).mean()


def log_spectral_distance(estimated_mag: torch.Tensor,
                          reference_mag: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Compute log spectral distance.
    
    Args:
        estimated_mag: Estimated magnitude spectrogram
        reference_mag: Reference magnitude spectrogram
        eps: Small epsilon for numerical stability
        
    Returns:
        lsd: Log spectral distance (lower is better)
    """
    log_est = torch.log(estimated_mag + eps)
    log_ref = torch.log(reference_mag + eps)
    
    squared_diff = (log_est - log_ref) ** 2
    lsd = torch.sqrt(torch.mean(squared_diff, dim=(-2, -1)))
    
    return lsd.mean()


class MetricsTracker:
    """
    Helper class to track and average metrics during training/validation.
    """
    
    def __init__(self, stem_names: List[str]):
        """
        Args:
            stem_names: List of stem names
        """
        self.stem_names = stem_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            stem: {'sdr': [], 'sir': [], 'sar': []}
            for stem in self.stem_names + ['average']
        }
        self.count = 0
    
    def update(self, metrics: Dict[str, Dict[str, float]]):
        """
        Update metrics with new batch results.
        
        Args:
            metrics: Dictionary from compute_separation_metrics
        """
        for stem_name in self.stem_names + ['average']:
            if stem_name in metrics:
                for metric_name in ['sdr', 'sir', 'sar']:
                    if metric_name in metrics[stem_name]:
                        self.metrics[stem_name][metric_name].append(
                            metrics[stem_name][metric_name]
                        )
        
        self.count += 1
    
    def compute_average(self) -> Dict[str, Dict[str, float]]:
        """
        Compute average metrics across all batches.
        
        Returns:
            avg_metrics: Dictionary of averaged metrics
        """
        avg_metrics = {}
        
        for stem_name in self.stem_names + ['average']:
            avg_metrics[stem_name] = {}
            for metric_name in ['sdr', 'sir', 'sar']:
                values = self.metrics[stem_name][metric_name]
                if values:
                    avg_metrics[stem_name][metric_name] = np.mean(values)
        
        return avg_metrics
    
    def get_summary_string(self) -> str:
        """
        Get a formatted string summary of metrics.
        
        Returns:
            summary: Formatted string with metric values
        """
        avg_metrics = self.compute_average()
        
        lines = []
        for stem_name in self.stem_names + ['average']:
            if stem_name in avg_metrics:
                metrics_str = ', '.join([
                    f"{metric}: {value:.2f} dB"
                    for metric, value in avg_metrics[stem_name].items()
                ])
                lines.append(f"{stem_name}: {metrics_str}")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing separation metrics...")
    
    # Create dummy data
    batch_size, channels, samples = 2, 2, 44100
    
    # Reference stems
    reference_stems = {
        'vocals': torch.randn(batch_size, channels, samples),
        'drums': torch.randn(batch_size, channels, samples),
        'bass': torch.randn(batch_size, channels, samples),
        'other': torch.randn(batch_size, channels, samples)
    }
    
    # Estimated stems (add some noise to reference)
    estimated_stems = {
        name: ref + 0.1 * torch.randn_like(ref)
        for name, ref in reference_stems.items()
    }
    
    # Compute metrics
    stem_names = ['vocals', 'drums', 'bass', 'other']
    metrics = compute_separation_metrics(estimated_stems, reference_stems, stem_names)
    
    print("\nMetrics per stem:")
    for stem_name, stem_metrics in metrics.items():
        if stem_name != 'average':
            print(f"\n{stem_name}:")
            for metric_name, value in stem_metrics.items():
                print(f"  {metric_name.upper()}: {value:.2f} dB")
    
    print("\nAverage metrics:")
    for metric_name, value in metrics['average'].items():
        print(f"  {metric_name.upper()}: {value:.2f} dB")
    
    # Test metrics tracker
    print("\n\nTesting MetricsTracker...")
    tracker = MetricsTracker(stem_names)
    
    # Simulate multiple batches
    for i in range(3):
        batch_metrics = compute_separation_metrics(estimated_stems, reference_stems, stem_names)
        tracker.update(batch_metrics)
    
    print("\nTracked metrics summary:")
    print(tracker.get_summary_string())