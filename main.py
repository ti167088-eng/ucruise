from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from services.assignment_service import AssignmentService # Corrected import path
from logger_config import get_logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssignmentRequest(BaseModel):
    source_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}")
def assign_drivers(source_id: str, parameter: int, string_param: str):
    logger = get_logger()  # Initialize the logger
    try:
        logger.info(f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}")

        # Use automatic API-based routing like run_and_view.py
        result = AssignmentService().run_assignment(source_id, parameter, string_param) # Instantiated the service and called run_assignment

        logger.info(f"🔍 Assignment result type: {type(result)}")
        logger.info(f"🔍 Assignment result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if result is None:
            logger.error("❌ Assignment returned None")
            return {"status": "false", "details": "Assignment returned None", "data": [], "parameter": parameter, "string_param": string_param}

        if not isinstance(result, dict):
            logger.error(f"❌ Assignment returned non-dict: {type(result)}")
            return {"status": "false", "details": f"Assignment returned {type(result)}", "data": [], "parameter": parameter, "string_param": string_param}

        # Ensure we have the required fields
        if "status" not in result:
            logger.warning("⚠️ No 'status' field in result, adding default")
            result["status"] = "true"
        
        if "data" not in result:
            logger.warning("⚠️ No 'data' field in result, adding empty array")
            result["data"] = []

        # Add parameter info if missing
        if "parameter" not in result:
            result["parameter"] = parameter
        if "string_param" not in result:
            result["string_param"] = string_param

        if result.get("status") == "true":
            logger.info(f"✅ Assignment successful. Routes: {len(result.get('data', []))}")
            routes_data = result.get("data", [])
            
            # Dump the full result to a file for debugging
            full_response_file = f"full_response_{source_id}_{parameter}.json"
            with open(full_response_file, "w") as f:
                import json
                json.dump(result, f, indent=2)
            logger.info(f"📁 Full response dumped to: {full_response_file}")
            
            # Also save routes for visualization compatibility
            if routes_data:
                with open("drivers_and_routes.json", "w") as f:
                    json.dump(routes_data, f, indent=2)
                logger.info("📁 Routes saved to drivers_and_routes.json")
        else:
            logger.error(f"❌ Assignment failed: {result.get('details', 'Unknown error')}")

        # Log the final response being sent
        logger.info(f"📤 Sending response: status={result.get('status')}, data_count={len(result.get('data', []))}")
        
        return result

    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        error_response = {
            "status": "false", 
            "details": f"Server error: {str(e)}", 
            "data": [], 
            "parameter": parameter, 
            "string_param": string_param
        }
        logger.info(f"📤 Sending error response: {error_response}")
        return error_response

@app.get("/routes")
async def get_routes():
    try:
        service = AssignmentService()
        result = service.run_assignment("UC_unify_dev", 1, "Evening%20shift")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routes/{optimization_mode}")
async def get_routes_optimized(optimization_mode: str):
    try:
        service = AssignmentService()
        result = service.run_assignment("UC_unify_dev", 1, "Evening%20shift", optimization_mode=optimization_mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)