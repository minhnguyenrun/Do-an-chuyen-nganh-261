import asyncio
from ..schemas.evidence import ParseEvidence


class ProbeScheduler:
    def __init__(self,max_glm_requests_per_page=30,max_in_flight=4):
        if max_glm_requests_per_page < 0 or max_in_flight < 1:
            raise ValueError('invalid inference limits')
        self.limit = min(30,max_glm_requests_per_page)
        self.used = 0
        self.semaphore = asyncio.Semaphore(max_in_flight)

    async def run(self,client,requests):
        # Reserve synchronously before yielding: concurrent run() calls share the same budget.
        reserved = []
        for request in requests:
            allowed = self.used < self.limit
            if allowed:
                self.used += 1
            reserved.append((request,allowed))
        async def execute(request,allowed):
            id,crop,prompt = request
            if not allowed:
                return ParseEvidence(hypothesis_id=id,parser='glm',prompt=prompt,content='',status='budget_exhausted')
            async with self.semaphore:
                try:
                    results = await client.recognize_batch([request])
                    if len(results) != 1 or results[0].hypothesis_id != id:
                        raise ValueError('GLM response IDs do not match requests')
                    return results[0]
                except Exception as exc:
                    return ParseEvidence(hypothesis_id=id,parser='glm',prompt=prompt,content='',status='error',error=str(exc))
        return await asyncio.gather(*(execute(req,allowed) for req,allowed in reserved))
