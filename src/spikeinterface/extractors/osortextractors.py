"""
SpikeInterface extractor for osort spike sorting results.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import warnings

try:
    import scipy.io as sio
except ImportError:
    sio = None

import spikeinterface as si
from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class OsortSortingExtractor(BaseSorting):
    """
    Extractor for osort spike sorting output.
    
    osort typically outputs results in .mat files containing spike times
    and cluster assignments.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the osort output file (.mat format)
    sampling_frequency : float
        Sampling frequency of the recording in Hz
    unit_ids : list, optional
        List of unit IDs to load. If None, all units are loaded
    keep_good_only : bool, default True
        If True, only load units marked as 'good'
    
    Returns
    -------
    sorting : OsortSortingExtractor
        The loaded sorting extractor
    """
    
    extractor_name = 'OsortSortingExtractor'
    installed = True
    is_writable = False
    mode = 'file'
    name = 'osort'

    def __init__(self, file_path: Union[str, Path], sampling_frequency: float, 
                 unit_ids: Optional[List] = None, keep_good_only: bool = True):
        
        if sio is None:
            raise ImportError("scipy is required for OsortSortingExtractor")
            
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")
            
        # Load the .mat file
        try:
            self._data = sio.loadmat(str(self._file_path), squeeze_me=True, struct_as_record=False)
        except Exception as e:
            raise ValueError(f"Could not load .mat file: {e}")
        
        self._sampling_frequency = float(sampling_frequency)
        self._keep_good_only = keep_good_only
        
        # Parse osort data structure
        self._parse_osort_data()
        
        # Filter units if specified
        if unit_ids is not None:
            available_units = set(self._unit_ids)
            requested_units = set(unit_ids)
            if not requested_units.issubset(available_units):
                missing = requested_units - available_units
                raise ValueError(f"Requested units not found: {missing}")
            self._unit_ids = list(requested_units)
        
        BaseSorting.__init__(self, sampling_frequency=self._sampling_frequency, unit_ids=self._unit_ids)
        
        # Add sorting segment
        sorting_segment = OsortSortingSegment(self._spike_times, self._spike_clusters, 
                                            self._unit_ids, self._sampling_frequency)
        self.set_property("artifact", np.array([u == 99999999 for u in self._unit_ids]))
        self.add_sorting_segment(sorting_segment)
        
        # Set properties if available
        self._set_unit_properties()
        
        # Add provenance
        self._kwargs = {
            'file_path': str(self._file_path.absolute()),
            'sampling_frequency': self._sampling_frequency,
            'keep_good_only': self._keep_good_only
        }

    def _parse_osort_data(self):
        """Parse the osort .mat file structure."""
        
        # Common osort output variable names to check
        possible_spike_time_vars = [ 'newSpikesNegative', 'newSpikesPositive']
        possible_cluster_vars = ['assignedNegative', 'assignedPositive']
        possible_quality_vars = ['cluster_group', 'group', 'quality', 'cluster_quality']
        
        # Find spike times
        spike_times_var = None
        for var in possible_spike_time_vars:
            if var in self._data:
                spike_times_var = var
                break
                
        if spike_times_var is None:
            raise ValueError(f"Could not find spike times variable. Available variables: {list(self._data.keys())}")
        
        self._spike_times = np.asarray(self._data[spike_times_var])
        
        # Find cluster assignments
        cluster_var = None
        for var in possible_cluster_vars:
            if var in self._data:
                cluster_var = var
                break
                
        if cluster_var is None:
            raise ValueError(f"Could not find cluster variable. Available variables: {list(self._data.keys())}")
            
        self._spike_clusters = np.asarray(self._data[cluster_var])
        
        # Ensure same length
        if len(self._spike_times) != len(self._spike_clusters):
            raise ValueError("Spike times and cluster assignments have different lengths")
        
        # Find quality information if available
        self._cluster_quality = None
        for var in possible_quality_vars:
            if var in self._data:
                self._cluster_quality = self._data[var]
                break
        
        # Get unique unit IDs
        unique_clusters = np.unique(self._spike_clusters)
        
        # Filter by quality if requested and available
        if self._keep_good_only and self._cluster_quality is not None:
            if isinstance(self._cluster_quality, dict):
                # Quality info as dictionary
                good_clusters = [clu for clu, qual in self._cluster_quality.items() 
                               if qual in ['good', 'Good', 1]]
            elif hasattr(self._cluster_quality, '__len__'):
                # Quality info as array
                good_mask = np.isin(self._cluster_quality, ['good', 'Good', 1])
                good_clusters = unique_clusters[good_mask] if len(self._cluster_quality) == len(unique_clusters) else unique_clusters
            else:
                good_clusters = unique_clusters
        else:
            good_clusters = unique_clusters
            
        # Remove noise cluster (usually -1 or 0)
        self._unit_ids = [int(clu) for clu in good_clusters if clu > 0]
        self._unit_ids.sort()

    def _set_unit_properties(self):
        """Set unit properties from osort data if available."""
        
        # Add quality information
        if self._cluster_quality is not None:
            quality_dict = {}
            for unit_id in self._unit_ids:
                if isinstance(self._cluster_quality, dict):
                    quality_dict[unit_id] = self._cluster_quality.get(unit_id, 'unknown')
                else:
                    quality_dict[unit_id] = 'good'  # Default if we filtered for good only
            self.set_property('quality', quality_dict)
        
        # Look for other common osort properties
        property_mappings = {
            'amplitude': ['amplitudes', 'amplitude', 'amp'],
            'firing_rate': ['firing_rate', 'fr', 'rate'],
            'isi_violation': ['isi_viol', 'isi_violation', 'contamination'],
            'isolation_distance': ['isolation_distance', 'iso_dist'],
            'l_ratio': ['l_ratio', 'L_ratio'],
            'd_prime': ['d_prime', 'dprime']
        }
        
        for prop_name, possible_vars in property_mappings.items():
            for var in possible_vars:
                if var in self._data:
                    prop_data = self._data[var]
                    if hasattr(prop_data, '__len__') and len(prop_data) >= len(self._unit_ids):
                        prop_dict = {unit_id: prop_data[i] for i, unit_id in enumerate(self._unit_ids)}
                        self.set_property(prop_name, prop_dict)
                    break

    @staticmethod
    def get_available_units(file_path: Union[str, Path]) -> List[int]:
        """
        Get list of available units in the osort file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the osort output file
            
        Returns
        -------
        unit_ids : list
            List of available unit IDs
        """
        if sio is None:
            raise ImportError("scipy is required for OsortSortingExtractor")
            
        data = sio.loadmat(str(file_path), squeeze_me=True)
        
        # Find cluster variable
        possible_vars = ['spike_clusters', 'spikeClusters', 'clusters', 'cluster_id', 'clu']
        clusters = None
        for var in possible_vars:
            if var in data:
                clusters = np.asarray(data[var])
                break
                
        if clusters is None:
            raise ValueError("Could not find cluster information in file")
            
        unique_clusters = np.unique(clusters)
        return [int(clu) for clu in unique_clusters if clu > 0]


class OsortSortingSegment(BaseSortingSegment):
    """Sorting segment for osort data."""
    
    def __init__(self, spike_times, spike_clusters, unit_ids, sampling_frequency):
        self._spike_times = spike_times
        self._spike_clusters = spike_clusters
        self._unit_ids = unit_ids
        self._sampling_frequency = sampling_frequency
        BaseSortingSegment.__init__(self)
    
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        """Get spike train for a specific unit."""
        
        # Get spikes for this unit
        unit_mask = self._spike_clusters == unit_id
        unit_spike_times = self._spike_times[unit_mask]
        
        # Convert to sample indices if needed
        if np.max(unit_spike_times) <= 1.0:
            # Assume times are in seconds, convert to samples
            spike_indices = np.round(unit_spike_times * self._sampling_frequency).astype(np.int64)
        else:
            # Assume already in samples
            spike_indices = unit_spike_times.astype(np.int64)
        
        # Apply frame limits
        if start_frame is not None:
            spike_indices = spike_indices[spike_indices >= start_frame]
        if end_frame is not None:
            spike_indices = spike_indices[spike_indices < end_frame]
            
        return np.sort(spike_indices)


# Register the extractor
read_osort = define_function_from_class(source_class=OsortSortingExtractor, name='read_osort')


def read_osort_sorting(file_path: Union[str, Path], sampling_frequency: float, 
                      unit_ids: Optional[List] = None, keep_good_only: bool = True) -> OsortSortingExtractor:
    """
    Read osort spike sorting results.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the osort output .mat file
    sampling_frequency : float
        Sampling frequency in Hz
    unit_ids : list, optional
        Specific unit IDs to load
    keep_good_only : bool, default True
        Only load units marked as good quality
        
    Returns
    -------
    sorting : OsortSortingExtractor
        Loaded sorting object
        
    Examples
    --------
    >>> sorting = read_osort_sorting('osort_output.mat', sampling_frequency=30000)
    >>> print(f"Loaded {len(sorting.unit_ids)} units")
    >>> spike_train = sorting.get_unit_spike_train(unit_id=1)
    """
    return OsortSortingExtractor(file_path, sampling_frequency, unit_ids, keep_good_only)


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the extractor
    
    # Load osort results
    # sorting = read_osort_sorting('path/to/osort_output.mat', sampling_frequency=30000)
    
    # Print basic info
    # print(f"Loaded sorting with {len(sorting.unit_ids)} units")
    # print(f"Unit IDs: {sorting.unit_ids}")
    # print(f"Sampling frequency: {sorting.sampling_frequency} Hz")
    
    # Get spike train for first unit
    # if len(sorting.unit_ids) > 0:
    #     unit_id = sorting.unit_ids[0]
    #     spike_train = sorting.get_unit_spike_train(unit_id)
    #     print(f"Unit {unit_id} has {len(spike_train)} spikes")
    
    # Check available properties
    # print(f"Available properties: {sorting.get_property_keys()}")
    
    pass