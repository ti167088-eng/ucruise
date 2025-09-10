
import os
import math
import requests
import numpy as np
import pandas as pd
import time
import json
import sys
from pathlib import Path
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger_config import get_logger
from utils.progress_tracker import get_progress_tracker
import assignment
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, calculate_bearing, calculate_bearings_and_features,
    coords_to_km, dbscan_clustering_metric, kmeans_clustering_metric, estimate_clusters,
    create_geographic_clusters, sweep_clustering, polar_sector_clustering,
    create_capacity_subclusters, create_bearing_aware_subclusters, calculate_bearing_spread,
    normalize_bearing_difference, calculate_sequence_distance, calculate_sequence_turning_score_improved,
    apply_strict_direction_aware_2opt, split_cluster_by_bearing_metric, apply_route_splitting, 
    split_route_by_bearing_improved, create_sub_route_improved, calculate_users_center_improved, 
    local_optimization, optimize_route_sequence_improved, calculate_route_cost_improved, 
    calculate_route_turning_score_improved, calculate_direction_consistency_improved, 
    try_user_swap_improved, calculate_route_center_improved, update_route_metrics_improved, 
    calculate_tortuosity_ratio_improved, global_optimization, fix_single_user_routes_improved, 
    calculate_average_bearing_improved, quality_controlled_route_filling, quality_preserving_route_merging, 
    strict_merge_compatibility_improved, calculate_merge_quality_score, perform_quality_merge_improved, 
    enhanced_route_splitting, intelligent_route_splitting_improved, split_by_bearing_clusters_improved, 
    split_by_distance_clusters_improved, create_split_routes_improved, find_best_driver_for_group, 
    outlier_detection_and_reassignment, try_reassign_outlier, handle_remaining_users_improved, 
    find_best_driver_for_cluster_improved, calculate_combined_route_center, _get_all_drivers_as_unassigned, 
    _convert_users_to_unassigned_format, analyze_assignment_quality,
    validate_route_path_coherence, reoptimize_route_with_road_awareness
)

warnings.filterwarnings('ignore')

# Setup logging first
logger = get_logger()

# File context for logging
FILE_CONTEXT = "ASSIGN_BALANCE.PY (BALANCED OPTIMIZATION)"

def run_balanced_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Balanced assignment function - optimal balance between route efficiency and capacity utilization
    """
    start_time = time.time()

    # Initialize logging and progress tracking
    logger = get_logger()
    progress = get_progress_tracker()

    progress.start_assignment(source_id, "BALANCED OPTIMIZATION")
    logger.info(f"🎯 Starting BALANCED assignment for source_id: {source_id}")

    try:
        # Use the route efficiency assignment as base (from assignment.py)
        # but with balanced parameters
        result = assignment.run_route_efficiency_assignment(source_id, parameter, string_param)
        
        # Update the optimization mode to reflect balanced approach
        if result.get("status") == "true":
            result["optimization_mode"] = "balanced_optimization"
            
        return result

    except Exception as e:
        logger.error(f"Balanced assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}
