"""
Algorithm-Level Caching System
Provides permanent caching of complete algorithm results with intelligent change detection
"""

import json
import hashlib
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from logger import get_logger

logger = get_logger()


class AlgorithmCache:
    """
    Manages permanent caching of algorithm results with data signature-based change detection
    """

    def __init__(self, db_name: str, algorithm_name: str, cache_dir: str = "routesCache"):
        """
        Initialize algorithm cache for specific company and algorithm

        Args:
            db_name: Company/database name (e.g., "UC_logisticllp")
            algorithm_name: Algorithm name (e.g., "base", "capacity", "balance", "safety", "road")
            cache_dir: Base cache directory
        """
        self.db_name = db_name
        self.algorithm_name = algorithm_name
        self.cache_dir = cache_dir

        # Create cache directory structure
        self.company_cache_dir = os.path.join(cache_dir, db_name)
        self.algorithm_cache_dir = os.path.join(self.company_cache_dir, algorithm_name)

        # Cache file paths
        self.result_cache_file = os.path.join(self.algorithm_cache_dir, "algorithm_result.json")
        self.signature_cache_file = os.path.join(self.algorithm_cache_dir, "data_signature.json")

        # Create directories if they don't exist
        os.makedirs(self.company_cache_dir, exist_ok=True)
        os.makedirs(self.algorithm_cache_dir, exist_ok=True)

        logger.info(f"🗂️ Algorithm cache initialized: {self.algorithm_cache_dir}")

    def generate_data_signature(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate signature from input data to detect changes

        Args:
            data: Input data dictionary from API
            parameters: Additional algorithm parameters

        Returns:
            Dictionary containing data signature components
        """
        try:
            # Extract users and drivers
            users = data.get('users', [])
            drivers_unassigned = data.get('driversUnassigned', [])
            drivers_assigned = data.get('driversAssigned', [])
            all_drivers = drivers_unassigned + drivers_assigned

            # User data signature
            user_count = len(users)
            user_data_for_hash = []
            for user in users:
                user_data_for_hash.append({
                    'id': str(user.get('id', user.get('sub_user_id', ''))),
                    'lat': float(user.get('latitude', 0.0)),
                    'lng': float(user.get('longitude', 0.0))
                })
            user_data_hash = self._hash_dict_list(user_data_for_hash)

            # Driver data signature
            driver_count = len(all_drivers)
            driver_data_for_hash = []
            for driver in all_drivers:
                driver_data_for_hash.append({
                    'id': str(driver.get('id', driver.get('sub_user_id', ''))),
                    'capacity': int(driver.get('capacity', 0)),
                    'lat': float(driver.get('latitude', 0.0)),
                    'lng': float(driver.get('longitude', 0.0))
                })
            driver_data_hash = self._hash_dict_list(driver_data_for_hash)

            # Configuration signature
            config_data = {
                'office_lat': float(data.get('company', {}).get('latitude', 0.0)),
                'office_lng': float(data.get('company', {}).get('longitude', 0.0)),
                'parameters': parameters or {}
            }
            config_hash = self._hash_string(json.dumps(config_data, sort_keys=True))

            # Combined signature
            signature_components = {
                'user_count': user_count,
                'driver_count': driver_count,
                'user_data_hash': user_data_hash,
                'driver_data_hash': driver_data_hash,
                'config_hash': config_hash,
                'combined_hash': self._hash_string(f"{user_count}_{user_data_hash}_{driver_count}_{driver_data_hash}_{config_hash}")
            }

            return signature_components

        except Exception as e:
            logger.error(f"Error generating data signature: {e}")
            return {}

    def get_cached_result(self, current_signature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached algorithm result if data signature matches

        Args:
            current_signature: Current data signature to compare

        Returns:
            Cached algorithm result if signatures match, None otherwise
        """
        try:
            # Check if cache files exist
            if not os.path.exists(self.result_cache_file) or not os.path.exists(self.signature_cache_file):
                logger.info("📭 No cached data found - will run algorithm")
                return None

            # Load cached signature
            with open(self.signature_cache_file, 'r', encoding='utf-8') as f:
                cached_signature = json.load(f)

            # Compare signatures
            current_hash = current_signature.get('combined_hash', '')
            cached_hash = cached_signature.get('combined_hash', '')

            if current_hash == cached_hash:
                logger.info(f"🎯 CACHE HIT: Data signature matches - returning cached result")
                logger.info(f"📊 Cached on: {cached_signature.get('timestamp', 'Unknown')}")

                # Load and return cached result
                with open(self.result_cache_file, 'r', encoding='utf-8') as f:
                    cached_result = json.load(f)

                # Add cache metadata
                cached_result['_cache_metadata'] = {
                    'cached': True,
                    'cache_timestamp': cached_signature.get('timestamp'),
                    'data_signature_match': True
                }

                return cached_result
            else:
                logger.info(f"🔄 CACHE MISS: Data signature changed - will recompute")
                logger.info(f"📊 Previous: {cached_hash[:16]}..., Current: {current_hash[:16]}...")
                return None

        except Exception as e:
            logger.error(f"Error checking cached result: {e}")
            return None

    def save_result_to_cache(self, result: Dict[str, Any], signature: Dict[str, Any]) -> bool:
        """
        Save algorithm result and signature to cache

        Args:
            result: Algorithm execution result to cache
            signature: Data signature for this result

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Prepare cache entry with timestamp
            timestamp = datetime.now().isoformat()

            # Save data signature
            signature_entry = signature.copy()
            signature_entry['timestamp'] = timestamp
            signature_entry['cache_version'] = '1.0'

            with open(self.signature_cache_file, 'w', encoding='utf-8') as f:
                json.dump(signature_entry, f, indent=2, ensure_ascii=False)

            # Save algorithm result
            result_entry = result.copy()
            result_entry['_cache_metadata'] = {
                'cached': True,
                'cache_timestamp': timestamp,
                'data_signature': signature,
                'cache_version': '1.0'
            }

            with open(self.result_cache_file, 'w', encoding='utf-8') as f:
                json.dump(result_entry, f, indent=2, ensure_ascii=False)

            # Get file sizes for logging
            result_size = os.path.getsize(self.result_cache_file)
            signature_size = os.path.getsize(self.signature_cache_file)

            logger.info(f"💾 CACHE UPDATE: Saved result ({result_size} bytes) + signature ({signature_size} bytes)")
            logger.info(f"📂 Cache location: {self.algorithm_cache_dir}")
            logger.info(f"⏰ Cached at: {timestamp}")

            return True

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False

    def clear_cache(self) -> bool:
        """
        Clear cached data for this algorithm

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            cleared_files = []

            if os.path.exists(self.result_cache_file):
                os.remove(self.result_cache_file)
                cleared_files.append(self.result_cache_file)

            if os.path.exists(self.signature_cache_file):
                os.remove(self.signature_cache_file)
                cleared_files.append(self.signature_cache_file)

            if cleared_files:
                logger.info(f"🗑️ Cache cleared: {', '.join(cleared_files)}")
            else:
                logger.info("📭 No cache files to clear")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache status

        Returns:
            Dictionary with cache status information
        """
        try:
            info = {
                'db_name': self.db_name,
                'algorithm_name': self.algorithm_name,
                'cache_dir': self.algorithm_cache_dir,
                'result_cache_exists': os.path.exists(self.result_cache_file),
                'signature_cache_exists': os.path.exists(self.signature_cache_file)
            }

            if info['result_cache_exists']:
                info['result_cache_size'] = os.path.getsize(self.result_cache_file)
                info['result_cache_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(self.result_cache_file)
                ).isoformat()

            if info['signature_cache_exists']:
                info['signature_cache_size'] = os.path.getsize(self.signature_cache_file)
                with open(self.signature_cache_file, 'r', encoding='utf-8') as f:
                    signature = json.load(f)
                    info['last_data_signature'] = signature.get('combined_hash', 'N/A')
                    info['last_timestamp'] = signature.get('timestamp', 'N/A')
                    info['user_count'] = signature.get('user_count', 0)
                    info['driver_count'] = signature.get('driver_count', 0)

            return info

        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'error': str(e)}

    def _hash_dict_list(self, dict_list: List[Dict[str, Any]]) -> str:
        """
        Hash a list of dictionaries consistently

        Args:
            dict_list: List of dictionaries to hash

        Returns:
            MD5 hash string
        """
        try:
            # Sort dictionaries by key to ensure consistent hashing
            sorted_dicts = []
            for d in dict_list:
                if isinstance(d, dict):
                    sorted_dict = {k: d[k] for k in sorted(d.keys())}
                    sorted_dicts.append(sorted_dict)
                else:
                    sorted_dicts.append(d)

            # Convert to JSON string with consistent formatting
            json_str = json.dumps(sorted_dicts, sort_keys=True, separators=(',', ':'))
            return hashlib.md5(json_str.encode('utf-8')).hexdigest()

        except Exception as e:
            logger.error(f"Error hashing dict list: {e}")
            return hashlib.md5(str(dict_list).encode('utf-8')).hexdigest()

    def _hash_string(self, text: str) -> str:
        """
        Hash a string using MD5

        Args:
            text: String to hash

        Returns:
            MD5 hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_algorithm_cache(db_name: str, algorithm_name: str) -> AlgorithmCache:
    """
    Factory function to get AlgorithmCache instance

    Args:
        db_name: Company/database name
        algorithm_name: Algorithm name

    Returns:
        AlgorithmCache instance
    """
    return AlgorithmCache(db_name, algorithm_name)