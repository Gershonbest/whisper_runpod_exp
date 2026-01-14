"""
Async request loop to test the Whisper transcription endpoint.

This script sends multiple concurrent requests to test the endpoint's
performance and reliability under load.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import json


# Configuration
ENDPOINT_URL = "https://fmoye410331wy7-8080.proxy.runpod.net/invocations"  # Change this to your endpoint URL
PING_URL = "https://fmoye410331wy7-8080.proxy.runpod.net/ping"  # Health check endpoint
NUM_REQUESTS = 50  # Number of concurrent requests
CONCURRENT_LIMIT = 50  # Maximum concurrent requests at once

# Test payload - audio transcription request
TEST_PAYLOAD = {
    "audio_url": "https://f616738f-backend.dataconect.com/api/v1/call-record-ext/documents/download/72681419-cb14-4a24-a301-6f947d5e50aa/CallRecord_1764594250722.mp3",
    "extra_data": {},
    "is_ml_agent": False,
    "dispatcher_endpoint": None  # Optional
}

# Alternative: ML agent request payload
ML_AGENT_PAYLOAD = {
    "transcript": [
        ["speaker1", "Hello, how are you?"],
        ["speaker2", "I'm doing well, thank you!"]
    ],
    "is_ml_agent": True,
    "extra_data": {
        "languages": {
            "defaultLanguage": "en"
        }
    },
    "dispatcher_endpoint": None
}


async def check_health(session: aiohttp.ClientSession) -> bool:
    """Check if the endpoint is healthy."""
    try:
        async with session.get(PING_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
            return response.status == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    payload: Dict[str, Any],
    use_ml_agent: bool = False
) -> Dict[str, Any]:
    """Send a single async request to the endpoint."""
    start_time = time.time()
    test_payload = ML_AGENT_PAYLOAD if use_ml_agent else payload
    
    try:
        async with session.post(
            ENDPOINT_URL,
            json=test_payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for transcription
        ) as response:
            elapsed_time = time.time() - start_time
            response_data = await response.json()
            
            return {
                "request_id": request_id,
                "status": response.status,
                "elapsed_time": elapsed_time,
                "success": response.status == 200,
                "response": response_data,
                "error": None
            }
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        return {
            "request_id": request_id,
            "status": None,
            "elapsed_time": elapsed_time,
            "success": False,
            "response": None,
            "error": "Request timeout"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "request_id": request_id,
            "status": None,
            "elapsed_time": elapsed_time,
            "success": False,
            "response": None,
            "error": str(e)
        }


async def run_concurrent_requests(
    num_requests: int,
    concurrent_limit: int,
    payload: Dict[str, Any],
    use_ml_agent: bool = False
) -> List[Dict[str, Any]]:
    """Run multiple concurrent requests with a limit on concurrency."""
    connector = aiohttp.TCPConnector(limit=concurrent_limit)
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minute total timeout
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Check health first
        print("Checking endpoint health...")
        if not await check_health(session):
            print("⚠️  Endpoint health check failed! Proceeding anyway...")
        else:
            print("✅ Endpoint is healthy")
        
        print(f"\n🚀 Sending {num_requests} requests with {concurrent_limit} concurrent limit...")
        print(f"📡 Endpoint: {ENDPOINT_URL}")
        print(f"📦 Request type: {'ML Agent' if use_ml_agent else 'Audio Transcription'}\n")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def bounded_request(request_id: int):
            async with semaphore:
                return await send_request(session, request_id, payload, use_ml_agent)
        
        # Create all tasks
        tasks = [bounded_request(i) for i in range(num_requests)]
        
        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "request_id": None,
                    "status": None,
                    "elapsed_time": 0,
                    "success": False,
                    "response": None,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results, total_time


def print_results(results: List[Dict[str, Any]], total_time: float):
    """Print test results in a formatted way."""
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n✅ Successful requests: {len(successful)}/{len(results)}")
    print(f"❌ Failed requests: {len(failed)}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Average time per request: {total_time/len(results):.2f}s")
    
    if successful:
        avg_success_time = sum(r["elapsed_time"] for r in successful) / len(successful)
        min_time = min(r["elapsed_time"] for r in successful)
        max_time = max(r["elapsed_time"] for r in successful)
        print(f"⚡ Average successful request time: {avg_success_time:.2f}s")
        print(f"🏃 Fastest request: {min_time:.2f}s")
        print(f"🐌 Slowest request: {max_time:.2f}s")
    
    if failed:
        print(f"\n❌ Failed Requests:")
        for result in failed:
            print(f"  Request {result.get('request_id', 'N/A')}: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("📋 Detailed Results:")
    print("="*80)
    
    for result in results:
        status_icon = "✅" if result.get("success") else "❌"
        print(f"\n{status_icon} Request {result.get('request_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Time: {result.get('elapsed_time', 0):.2f}s")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        if result.get("response"):
            # Print a summary of the response
            response = result["response"]
            if isinstance(response, dict):
                if "data" in response:
                    data = response["data"]
                    print(f"   Response: text length={len(data.get('text', ''))}, "
                          f"language={data.get('language', 'N/A')}, "
                          f"duration={data.get('duration', 'N/A')}")
                else:
                    print(f"   Response keys: {list(response.keys())}")


async def main():
    """Main function to run the async test loop."""
    print("🧪 Whisper Endpoint Async Test Suite")
    print("="*80)
    
    # Test with audio transcription
    print("\n📝 Test 1: Audio Transcription Requests")
    results, total_time = await run_concurrent_requests(
        num_requests=NUM_REQUESTS,
        concurrent_limit=CONCURRENT_LIMIT,
        payload=TEST_PAYLOAD,
        use_ml_agent=False
    )
    print_results(results, total_time)
    
    # Uncomment to test ML agent requests as well
    # print("\n\n🤖 Test 2: ML Agent Requests")
    # results_ml, total_time_ml = await run_concurrent_requests(
    #     num_requests=5,
    #     concurrent_limit=3,
    #     payload=ML_AGENT_PAYLOAD,
    #     use_ml_agent=True
    # )
    # print_results(results_ml, total_time_ml)


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())

