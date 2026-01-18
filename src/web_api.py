"""
Web API Module with CORS Support

This module provides a REST API for the Trakt OCR application with CORS support.
"""

import json
import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_folder="../web", static_url_path="")
CORS(
    app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"]
)

logger = logging.getLogger(__name__)

# Application state
app_state = {
    "status": "stopped",
    "frame_count": 0,
    "detection_count": 0,
    "last_update": None,
    "camera_info": None,
    "cameras": {},
    "recent_detections": [],
}

# Database and Alert manager instances (set externally)
_database = None
_alert_manager = None
_multi_camera_manager = None


def set_database(db):
    """Set the database manager instance."""
    global _database
    _database = db


def set_alert_manager(am):
    """Set the alert manager instance."""
    global _alert_manager
    _alert_manager = am


def set_multi_camera_manager(mcm):
    """Set the multi-camera manager instance."""
    global _multi_camera_manager
    _multi_camera_manager = mcm


@app.route("/")
def index():
    """Serve the main web interface."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def get_status():
    """Get application status."""
    data = {
        "status": app_state["status"],
        "frame_count": app_state["frame_count"],
        "detection_count": app_state["detection_count"],
        "last_update": app_state["last_update"],
        "camera_info": app_state["camera_info"],
    }

    # Add multi-camera status if available
    if _multi_camera_manager:
        data["cameras"] = _multi_camera_manager.get_statistics()

    return jsonify({"success": True, "data": data})


@app.route("/api/detections")
def get_detections():
    """Get recent OCR detections."""
    limit = request.args.get("limit", 50, type=int)
    camera_id = request.args.get("camera_id")

    # Try database first
    if _database:
        detections = _database.get_detections(camera_id=camera_id, limit=limit)
        return jsonify({"success": True, "data": detections, "count": len(detections)})

    # Fall back to in-memory state
    detections = app_state["recent_detections"][-limit:]
    return jsonify({"success": True, "data": detections, "count": len(detections)})


@app.route("/api/camera/info")
def get_camera_info():
    """Get camera information."""
    return jsonify({"success": True, "data": app_state["camera_info"]})


@app.route("/api/cameras")
def get_cameras():
    """Get all cameras status."""
    if _multi_camera_manager:
        cameras = _multi_camera_manager.get_camera_status()
        return jsonify({"success": True, "data": cameras})

    # Single camera mode
    return jsonify({"success": True, "data": {"default": app_state["camera_info"]}})


@app.route("/api/cameras/<camera_id>")
def get_camera(camera_id):
    """Get specific camera status."""
    if _multi_camera_manager:
        status = _multi_camera_manager.get_camera_status(camera_id)
        if status:
            return jsonify({"success": True, "data": status})
        return jsonify({"success": False, "error": "Camera not found"}), 404

    return jsonify({"success": False, "error": "Multi-camera mode not enabled"}), 400


@app.route("/api/results")
def get_results():
    """Get saved OCR results from files."""
    import heapq

    results_dir = os.path.join(os.path.dirname(__file__), "..", "output", "results")
    results = []

    if os.path.exists(results_dir):
        # Get json files and use heapq for efficient top-N selection
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        # Get the 100 most recent files (largest names = most recent by timestamp)
        recent_files = heapq.nlargest(100, json_files)

        for filename in recent_files:
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logger.warning(f"Failed to read result file {filename}: {e}")

    return jsonify({"success": True, "data": results, "count": len(results)})


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {"success": True, "status": "healthy", "timestamp": datetime.now().isoformat()}
    )


# Database endpoints
@app.route("/api/database/detections")
def get_database_detections():
    """Get detections from database."""
    if not _database:
        return jsonify({"success": False, "error": "Database not enabled"}), 400

    camera_id = request.args.get("camera_id")
    start_time = request.args.get("start_time")
    end_time = request.args.get("end_time")
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    detections = _database.get_detections(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return jsonify({"success": True, "data": detections, "count": len(detections)})


@app.route("/api/database/statistics")
def get_database_statistics():
    """Get database statistics."""
    if not _database:
        return jsonify({"success": False, "error": "Database not enabled"}), 400

    camera_id = request.args.get("camera_id")
    stats = _database.get_statistics(camera_id=camera_id)
    return jsonify({"success": True, "data": stats})


# Alert endpoints
@app.route("/api/alerts")
def get_alerts():
    """Get alerts from the system."""
    if not _alert_manager:
        return jsonify({"success": False, "error": "Alert system not enabled"}), 400

    if _database:
        camera_id = request.args.get("camera_id")
        status = request.args.get("status")
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)

        alerts = _database.get_alerts(
            camera_id=camera_id, status=status, limit=limit, offset=offset
        )
        return jsonify({"success": True, "data": alerts, "count": len(alerts)})

    # Fall back to recent alerts from alert manager
    alerts = _alert_manager.get_recent_alerts(limit=50)
    return jsonify({"success": True, "data": alerts, "count": len(alerts)})


@app.route("/api/alerts/<int:alert_id>/status", methods=["POST"])
def update_alert_status(alert_id):
    """Update alert status."""
    if not _database:
        return jsonify({"success": False, "error": "Database not enabled"}), 400

    data = request.get_json()
    if not data or "status" not in data:
        return jsonify({"success": False, "error": "Status is required"}), 400

    status = data["status"]
    if status not in ["pending", "acknowledged", "resolved", "dismissed"]:
        return jsonify({"success": False, "error": "Invalid status"}), 400

    success = _database.update_alert_status(alert_id, status)
    if success:
        return jsonify({"success": True, "message": "Alert status updated"})
    return jsonify({"success": False, "error": "Alert not found"}), 404


@app.route("/api/alerts/patterns")
def get_alert_patterns():
    """Get configured alert patterns."""
    if not _alert_manager:
        return jsonify({"success": False, "error": "Alert system not enabled"}), 400

    patterns = _alert_manager.get_patterns()
    return jsonify({"success": True, "data": patterns, "count": len(patterns)})


@app.route("/api/alerts/patterns/<name>/enabled", methods=["POST"])
def set_pattern_enabled(name):
    """Enable or disable an alert pattern."""
    if not _alert_manager:
        return jsonify({"success": False, "error": "Alert system not enabled"}), 400

    data = request.get_json()
    if not data or "enabled" not in data:
        return jsonify({"success": False, "error": "Enabled flag is required"}), 400

    success = _alert_manager.set_pattern_enabled(name, data["enabled"])
    if success:
        return jsonify({"success": True, "message": "Pattern updated"})
    return jsonify({"success": False, "error": "Pattern not found"}), 404


def update_state(
    frame_count=None,
    detection_count=None,
    detections=None,
    camera_info=None,
    status=None,
):
    """Update application state from the main OCR app."""
    if frame_count is not None:
        app_state["frame_count"] = frame_count
    if detection_count is not None:
        app_state["detection_count"] = detection_count
    if detections is not None:
        app_state["recent_detections"].extend(detections)
        # Keep only the last 1000 detections
        if len(app_state["recent_detections"]) > 1000:
            app_state["recent_detections"] = app_state["recent_detections"][-1000:]
    if camera_info is not None:
        app_state["camera_info"] = camera_info
    if status is not None:
        app_state["status"] = status
    app_state["last_update"] = datetime.now().isoformat()


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask web server.

    Args:
        host: Host to bind to. Default is '127.0.0.1' (localhost only) for security.
              Set to '0.0.0.0' to allow external connections (not recommended without proper security).
        port: Port to bind to. Default is 5000.
        debug: Enable debug mode. Default is False.
    """
    logger.info(f"Starting web API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Only enable debug mode if explicitly set via environment variable
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    # Allow host configuration via environment variable for production deployments
    # Default is 127.0.0.1 (localhost only) for security
    host = os.environ.get("FLASK_HOST", "127.0.0.1")

    # Validate port is numeric and within valid range
    try:
        port = int(os.environ.get("FLASK_PORT", "5000"))
        if not (1 <= port <= 65535):
            logger.warning(f"Invalid port {port}, using default 5000")
            port = 5000
    except ValueError:
        logger.warning("Invalid FLASK_PORT value, using default 5000")
        port = 5000

    run_server(host=host, port=port, debug=debug_mode)
