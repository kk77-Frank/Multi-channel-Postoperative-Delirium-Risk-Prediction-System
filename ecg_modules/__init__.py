#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG Module Initialization File
"""

# Version information
__version__ = '1.0.0'

# Import modules in dependency order
from ecg_modules.utils import find_ecg_files, detect_signal_quality
from ecg_modules.preprocessing import SignalPreprocessor
# Lazy import other modules to avoid circular references
import ecg_modules.feature_extraction
import ecg_modules.feature_selection
import ecg_modules.visualization
import ecg_modules.model_building

# Export commonly used classes for backward compatibility
FeatureExtractor = ecg_modules.feature_extraction.FeatureExtractor
FeatureSelector = ecg_modules.feature_selection.FeatureSelector
Visualizer = ecg_modules.visualization.Visualizer
ModelBuilder = ecg_modules.model_building.ModelBuilder
build_full_delirium_model = ecg_modules.model_building.build_full_delirium_model
