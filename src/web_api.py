"""
Web API Module with CORS Support

This module provides a REST API for the Trakt OCR application with CORS support.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
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
    "recent_detections": [],
}


@app.route("/")
def index():
    """Serve the main web interface."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def get_status():
    """Get application status."""
    return jsonify(
        {
            "success": True,
            "data": {
                "status": app_state["status"],
                "frame_count": app_state["frame_count"],
                "detection_count": app_state["detection_count"],
                "last_update": app_state["last_update"],
                "camera_info": app_state["camera_info"],
            },
        }
    )


@app.route("/api/detections")
def get_detections():
    """Get recent OCR detections."""
    limit = request.args.get("limit", 50, type=int)
    detections = app_state["recent_detections"][-limit:]
    return jsonify({"success": True, "data": detections, "count": len(detections)})


@app.route("/api/camera/info")
def get_camera_info():
    """Get camera information."""
    return jsonify({"success": True, "data": app_state["camera_info"]})


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


def run_server(host="0.0.0.0", port=5000, debug=False):  # nosec B104
    """Run the Flask web server."""
    logger.info(f"Starting web API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    # Only enable debug mode if explicitly set via environment variable
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    run_server(debug=debug_mode)
