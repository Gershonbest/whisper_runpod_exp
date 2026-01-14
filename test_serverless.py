import asyncio
import aiohttp
import json

async def runpod_request(session, api_key: str, payload: dict, i: int):
    url = "https://api.runpod.ai/v2/kxp1ji51zyf46e/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request_payload = {"input": payload}

    try:
        async with session.post(url, headers=headers, json=request_payload, timeout=300) as response:
            data = await response.json()
            print(f"\n{'='*80}")
            print(f"Request {i + 1} -> Status: {response.status}")
            print(f"{'='*80}")
            print(f"Response:")
            print(json.dumps(data, indent=2))
            print(f"{'='*80}\n")
            return data
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Request {i + 1} -> Error: {str(e)}")
        print(f"{'='*80}\n")
        return {"error": str(e)}

async def runpod_request_50_async(api_key: str, payload: dict):
    async with aiohttp.ClientSession() as session:
        tasks = [
            runpod_request(session, api_key, payload, i)
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
    return results

# Entry point
if __name__ == "__main__":
    import os

    API_KEY = ""
    PAYLOAD = {
        "audio_url": "https://f616738f-backend.dataconect.com/api/v1/call-record-ext/documents/download/72681419-cb14-4a24-a301-6f947d5e50aa/CallRecord_1764594250722.mp3",
        "extra_data": {},
        "is_ml_agent": False,
        "dispatcher_endpoint": None
    }

    print("🚀 Starting 50 async requests to RunPod serverless endpoint...")
    print(f"📡 Endpoint: https://api.runpod.ai/v2/kxp1ji51zyf46e/run")
    print(f"📦 Payload: Audio transcription request\n")
    
    results = asyncio.run(runpod_request_50_async(API_KEY, PAYLOAD))
    
    print(f"\n{'='*80}")
    print(f"✅ Completed {len(results)} requests")
    print(f"{'='*80}")
    
    # Summary statistics
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n📊 Sample successful response structure:")
        if successful[0].get("output"):
            print(json.dumps(successful[0]["output"], indent=2))
        elif successful[0].get("data"):
            print(json.dumps(successful[0]["data"], indent=2))
        else:
            print(json.dumps(successful[0], indent=2))