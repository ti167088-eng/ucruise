import logging
import os
from datetime import datetime
import json
import glob

def clear_logs(log_dir="logs"):
    """Clear all existing log files before starting a new session"""
    if os.path.exists(log_dir):
        log_files = glob.glob(os.path.join(log_dir, "assignment_*.log"))
        removed_count = 0
        for log_file in log_files:
            try:
                os.remove(log_file)
                removed_count += 1
            except OSError as e:
                print(f"Warning: Could not remove {log_file}: {e}")

        if removed_count > 0:
            print(f"🧹 Cleared {removed_count} old log files from {log_dir}/")
        else:
            print(f"📂 No existing log files found in {log_dir}/")
    else:
        print(f"📂 Log directory {log_dir}/ doesn't exist yet")

class RouteAssignmentLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp for this session with milliseconds for uniqueness
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Setup main logger
        self.logger = logging.getLogger('route_assignment')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create file handler with UTF-8 encoding
        log_file = os.path.join(log_dir, f"assignment_{self.session_timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # All tracking will go to the main log file
        self.tracking_file = log_file

        self.log_session_start()

    def log_session_start(self):
        self.logger.info("="*80)
        self.logger.info("ROUTE ASSIGNMENT SESSION STARTED")
        self.logger.info(f"Session ID: {self.session_timestamp}")
        self.logger.info("="*80)

    def info(self, message):
        """Add info method for compatibility"""
        self.logger.info(message)

    def warning(self, message):
        """Add warning method for compatibility"""
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        """Add error method for compatibility"""
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message):
        """Add critical method for compatibility"""
        self.logger.critical(message)

    def debug(self, message):
        """Add debug method for compatibility"""
        self.logger.debug(message)

    def log_data_validation(self, users_count, drivers_count, office_coords):
        self.logger.info(f"DATA VALIDATION - Users: {users_count}, Drivers: {drivers_count}")
        self.logger.info(f"Office coordinates: {office_coords}")

    def log_clustering_decision(self, method, user_count, cluster_count, details):
        self.logger.info(f"CLUSTERING - Method: {method}, Users: {user_count}, Clusters: {cluster_count}")
        self.logger.debug(f"Clustering details: {details}")

    def log_route_creation(self, driver_id, users, reason, quality_metrics):
        self.logger.info(f"ROUTE CREATED - Driver: {driver_id}, Users: {len(users)}")
        self.logger.info(f"Creation reason: {reason}")
        for user in users:
            self.logger.debug(f"  User {user.get('user_id', 'N/A')}")
        self.logger.debug(f"Quality metrics: {quality_metrics}")

    def log_route_rejection(self, driver_id, users, reason):
        self.logger.warning(f"ROUTE REJECTED - Driver: {driver_id}, Reason: {reason}")
        for user in users:
            self.logger.debug(f"  Rejected user {user.get('user_id', 'N/A')}")

    def log_user_assignment(self, user_id, driver_id, route_details):
        self.logger.info(f"USER ASSIGNED - User: {user_id} -> Driver: {driver_id}")
        self.logger.debug(f"Route details: {route_details}")

    def log_user_unassigned(self, user_id, reason, attempted_drivers):
        self.logger.warning(f"USER UNASSIGNED - User: {user_id}, Reason: {reason}")
        self.logger.debug(f"Attempted drivers: {attempted_drivers}")

        # Additional tracking info in main log
        self.logger.info(f"TRACKING | UNASSIGNED_USER | {user_id} | {reason}")

    def log_driver_unused(self, driver_id, reason, capacity, location):
        self.logger.warning(f"DRIVER UNUSED - Driver: {driver_id}, Reason: {reason}")
        self.logger.debug(f"Capacity: {capacity}, Location: {location}")

        # Additional tracking info in main log
        self.logger.info(f"TRACKING | UNUSED_DRIVER | {driver_id} | {reason} | {capacity} | {location}")

    def log_optimization_step(self, step_name, before_state, after_state, changes):
        self.logger.info(f"OPTIMIZATION - {step_name}")
        self.logger.debug(f"Before: {before_state}")
        self.logger.debug(f"After: {after_state}")
        self.logger.info(f"Changes made: {changes}")

    def log_final_summary(self, total_users, assigned_users, unassigned_users, 
                         total_drivers, used_drivers, unused_drivers, routes):
        self.logger.info("="*80)
        self.logger.info("FINAL ASSIGNMENT SUMMARY")
        self.logger.info(f"Users: {assigned_users}/{total_users} assigned ({len(unassigned_users)} unassigned)")
        self.logger.info(f"Drivers: {used_drivers}/{total_drivers} used ({len(unused_drivers)} unused)")
        self.logger.info(f"Routes created: {len(routes)}")

        # Detailed unassigned analysis
        if unassigned_users:
            self.logger.warning(f"UNASSIGNED USERS ANALYSIS ({len(unassigned_users)} users):")
            for user in unassigned_users:
                self.logger.warning(f"  User {user.get('user_id', 'N/A')}")

        if unused_drivers:
            self.logger.warning(f"UNUSED DRIVERS ANALYSIS ({len(unused_drivers)} drivers):")
            for driver in unused_drivers:
                self.logger.warning(f"  Driver {driver.get('driver_id', 'N/A')} capacity {driver.get('capacity', 'N/A')}")

        self.logger.info("="*80)

    def log_accounting_check(self, total_api_users, final_assigned, final_unassigned, discrepancy):
        """Log comprehensive user accounting check"""
        self.logger.critical("USER ACCOUNTING CHECK")
        self.logger.critical(f"API Users: {total_api_users}")
        self.logger.critical(f"Final Assigned: {final_assigned}")

        # Handle final_unassigned as either int (count) or list (actual users)
        if isinstance(final_unassigned, list):
            unassigned_count = len(final_unassigned)
            self.logger.critical(f"Final Unassigned: {unassigned_count}")
            total_accounted = final_assigned + unassigned_count
        else:
            unassigned_count = final_unassigned
            self.logger.critical(f"Final Unassigned: {unassigned_count}")
            total_accounted = final_assigned + unassigned_count

        self.logger.critical(f"Total Accounted: {total_accounted}")
        self.logger.critical(f"Discrepancy: {discrepancy}")

        if discrepancy != 0:
            self.logger.critical(f"WARNING: {discrepancy} users unaccounted for!")
        else:
            self.logger.critical("✅ User accounting verified")

    def log_route_verification(self, routes):
        """Log detailed route verification for debugging map mismatches"""
        self.logger.info("🔍 DETAILED ROUTE VERIFICATION:")
        self.logger.info("="*60)

        for i, route in enumerate(routes, 1):
            self.logger.info(f"Route {i} - Driver {route['driver_id']}:")
            self.logger.info(f"  📍 Driver coordinates: ({route['latitude']:.6f}, {route['longitude']:.6f})")
            self.logger.info(f"  🚗 Vehicle type/capacity: {route['vehicle_type']}")
            self.logger.info(f"  🆔 Vehicle ID: {route.get('vehicle_id', 'N/A')}")
            self.logger.info(f"  👥 Number of users: {len(route['assigned_users'])}")

            if route['assigned_users']:
                self.logger.info(f"  📋 User details:")
                for j, user in enumerate(route['assigned_users'], 1):
                    office_dist = user.get('office_distance', 'N/A')
                    self.logger.info(f"    {j}. User {user['user_id']}")
                    self.logger.info(f"       📍 Coordinates: ({user['lat']:.6f}, {user['lng']:.6f})")
                    self.logger.info(f"       🏢 Office distance: {office_dist}km")
                    self.logger.info(f"       👤 Name: {user.get('first_name', 'N/A')}")
                    self.logger.info(f"       ✉️ Email: {user.get('email', 'N/A')}")
            else:
                self.logger.warning(f"  ❌ NO USERS ASSIGNED TO THIS ROUTE!")

            utilization = len(route['assigned_users']) / route['vehicle_type'] if route['vehicle_type'] > 0 else 0
            self.logger.info(f"  📊 Utilization: {utilization*100:.1f}%")
            self.logger.info(f"  📏 Total distance: {route.get('total_distance', 'N/A')}km")
            self.logger.info("")

    def log_driver_assignment_summary(self, all_drivers, used_driver_ids, unused_drivers):
        """Log comprehensive driver assignment summary"""
        self.logger.info("🚗 DRIVER ASSIGNMENT SUMMARY:")
        self.logger.info("="*60)

        self.logger.info(f"📊 Driver statistics:")
        self.logger.info(f"  Total drivers available: {len(all_drivers)}")
        self.logger.info(f"  Drivers used: {len(used_driver_ids)}")
        self.logger.info(f"  Drivers unused: {len(unused_drivers)}")

        self.logger.info(f"📋 Used driver details:")
        for driver_id in sorted(list(used_driver_ids)):
            # Find the driver details
            driver_details = None
            for driver in all_drivers:
                if str(driver['id']) == driver_id:
                    driver_details = driver
                    break

            if driver_details:
                self.logger.info(f"  ✅ Driver {driver_id}: capacity {driver_details['capacity']}, location ({driver_details['latitude']:.6f}, {driver_details['longitude']:.6f})")
            else:
                self.logger.info(f"  ❓ Driver {driver_id}: details not found")

        if unused_drivers:
            self.logger.info(f"📋 Unused driver details:")
            for driver in unused_drivers:
                self.logger.info(f"  ❌ Driver {driver['driver_id']}: capacity {driver['capacity']}, reason: {driver.get('reason', 'Unknown')}")

# Global logger instance and session tracking
route_logger = None
current_session_id = None

def get_logger():
    global route_logger
    if route_logger is None:
        route_logger = RouteAssignmentLogger()
    return route_logger

def reset_logger():
    """Reset logger for new session - only if not in an active session"""
    global route_logger, current_session_id

    # Don't reset if we're in the middle of a session
    if current_session_id is not None:
        return route_logger

    if route_logger is not None:
        # Close existing handlers before resetting
        for handler in route_logger.logger.handlers[:]:
            handler.close()
            route_logger.logger.removeHandler(handler)
    route_logger = None
    return None

def start_session():
    """Start a new logging session"""
    global current_session_id

    # Clear old logs before starting new session
    clear_logs()

    current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return get_logger()

def end_session():
    """End the current logging session"""
    global current_session_id
    current_session_id = None