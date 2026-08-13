import random
import hashlib
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

class Chunk:
    """表示一个文本块"""
    def __init__(self, chunk_id: str, content: str = None):
        self.chunk_id = chunk_id
        self.content = content or f"Content of {chunk_id}"
        self.kv_size_mb = random.uniform(0.5, 2.0)  # KVCache大小 (MB)
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        return self.chunk_id == other.chunk_id
    
    def __repr__(self):
        return f"Chunk({self.chunk_id})"

class Query:
    """表示一个用户查询"""
    def __init__(self, query_id: int, chunks: List[Chunk]):
        self.query_id = query_id
        self.chunks = chunks  # 包含system prompt

class BaselinePrefixCache:
    """基线系统：标准前缀缓存（无 Alternative Path）"""
    def __init__(self):
        # key: prefix的hash, value: (prefix_chunks, total_kv_size)
        self.cache: Dict[str, Tuple[List[str], float]] = {}
        self.total_storage_mb = 0.0
        self.chunk_recompute_count = defaultdict(int)  # 记录每个chunk被计算的次数
        
        # 总体缓存命中率统计
        self.total_chunk_accesses = 0  # 总的chunk访问次数
        self.cache_hits = 0             # 缓存命中次数
        self.cache_misses = 0           # 缓存未命中次数
    
    def _get_prefix_hash(self, chunks: List[Chunk]) -> str:
        """计算prefix的hash"""
        prefix_str = "+".join([c.chunk_id for c in chunks])
        return hashlib.md5(prefix_str.encode()).hexdigest()
    
    def process_query(self, query: Query) -> Dict:
        """
        处理查询，使用标准prefix cache
        逐个chunk尝试匹配最长前缀
        """
        chunks = query.chunks
        matched_chunks = []
        storage_added = 0.0
        chunks_computed = []
        
        # 逐个chunk处理
        for i in range(len(chunks)):
            self.total_chunk_accesses += 1  # 每访问一个chunk，计数+1
            
            current_prefix = chunks[:i+1]
            prefix_hash = self._get_prefix_hash(current_prefix)
            
            if prefix_hash in self.cache:
                # 前缀命中，继续
                self.cache_hits += 1
                matched_chunks = current_prefix
            else:
                # 前缀未命中，需要计算当前chunk
                self.cache_misses += 1
                
                # 但已经匹配的部分不需要重新计算
                chunks_to_compute = current_prefix[len(matched_chunks):]
                
                for chunk in chunks_to_compute:
                    self.chunk_recompute_count[chunk.chunk_id] += 1
                    storage_added += chunk.kv_size_mb
                    chunks_computed.append(chunk.chunk_id)
                
                # 缓存当前prefix
                self.cache[prefix_hash] = (
                    [c.chunk_id for c in current_prefix],
                    sum(c.kv_size_mb for c in current_prefix)
                )
                matched_chunks = current_prefix
        
        self.total_storage_mb += storage_added
        
        return {
            'storage_added': storage_added,
            'chunks_computed': chunks_computed,
        }

class FusionRAGCache:
    """FusionRAG：带 Alternative Path 的缓存"""
    def __init__(self):
        # key: prefix的hash, value: (prefix_chunks, total_kv_size)
        self.cache: Dict[str, Tuple[List[str], float]] = {}
        self.total_storage_mb = 0.0
        self.chunk_recompute_count = defaultdict(int)
        self.alternative_path_hits = 0
        self.total_alternative_path_attempts = 0
        
        # 总体缓存命中率统计
        self.total_chunk_accesses = 0  # 总的chunk访问次数
        self.prefix_cache_hits = 0      # prefix cache命中次数
        self.alternative_path_cache_hits = 0  # alternative path命中次数
        self.cache_misses = 0           # 缓存未命中次数（需要计算）
    
    def _get_prefix_hash(self, chunks: List[Chunk]) -> str:
        """计算prefix的hash"""
        prefix_str = "+".join([c.chunk_id for c in chunks])
        return hashlib.md5(prefix_str.encode()).hexdigest()
    
    def _try_alternative_paths(self, prefix: List[Chunk], current_chunk: Chunk) -> Optional[str]:
        """
        对于当前文本块，尝试alternative paths
        prefix: 当前chunk之前的所有chunks (不包括当前chunk)
        current_chunk: 当前要匹配的chunk
        
        策略：从最长的prefix开始，每次删掉最后一个chunk，
        然后和current_chunk拼接，尝试匹配
        """
        self.total_alternative_path_attempts += 1
        
        # 从完整prefix开始尝试
        for length in range(len(prefix), 0, -1):
            candidate_prefix = prefix[:length]
            candidate_path = candidate_prefix + [current_chunk]
            path_hash = self._get_prefix_hash(candidate_path)
            
            if path_hash in self.cache:
                self.alternative_path_hits += 1
                return path_hash
        
        return None
    
    def process_query(self, query: Query) -> Dict:
        """
        处理查询，使用Alternative Path
        """
        chunks = query.chunks
        matched_chunks = []
        storage_added = 0.0
        chunks_computed = []
        alt_path_hits_this_query = 0
        
        # 逐个chunk处理
        for i in range(len(chunks)):
            self.total_chunk_accesses += 1  # 每访问一个chunk，计数+1
            
            current_prefix = chunks[:i+1]
            prefix_hash = self._get_prefix_hash(current_prefix)
            
            if prefix_hash in self.cache:
                # 完整前缀命中
                self.prefix_cache_hits += 1
                matched_chunks = current_prefix
            else:
                # 前缀未命中
                # 对于已经匹配的部分，不需要重新处理
                # 对于新的chunks（从len(matched_chunks)到i），需要处理
                
                start_idx = len(matched_chunks)
                for j in range(start_idx, i + 1):
                    current_chunk = chunks[j]
                    prefix_before_current = chunks[:j]  # 当前chunk之前的所有chunks
                    
                    # 尝试alternative path匹配当前chunk
                    matched_hash = None
                    if len(prefix_before_current) > 0:
                        matched_hash = self._try_alternative_paths(prefix_before_current, current_chunk)
                    
                    if matched_hash:
                        # Alternative path命中！
                        alt_path_hits_this_query += 1
                        self.alternative_path_cache_hits += 1
                        # 不需要计算这个chunk，它已经被缓存了
                    else:
                        # 没有找到匹配，需要计算当前chunk
                        self.cache_misses += 1
                        self.chunk_recompute_count[current_chunk.chunk_id] += 1
                        storage_added += current_chunk.kv_size_mb
                        chunks_computed.append(current_chunk.chunk_id)
                        
                        # 缓存 [prefix_before_current + current_chunk]
                        new_prefix = prefix_before_current + [current_chunk]
                        new_hash = self._get_prefix_hash(new_prefix)
                        self.cache[new_hash] = (
                            [c.chunk_id for c in new_prefix],
                            sum(c.kv_size_mb for c in new_prefix)
                        )
                
                # 缓存完整的当前prefix
                self.cache[prefix_hash] = (
                    [c.chunk_id for c in current_prefix],
                    sum(c.kv_size_mb for c in current_prefix)
                )
                matched_chunks = current_prefix
        
        self.total_storage_mb += storage_added
        
        return {
            'storage_added': storage_added,
            'chunks_computed': chunks_computed,
            'alt_path_hits': alt_path_hits_this_query,
        }

class RAGSimulator:
    """RAG场景模拟器"""
    def __init__(self, 
                 num_kb_chunks: int = 100,
                 num_queries: int = 1000,
                 chunks_per_query: Tuple[int, int] = (3, 8),
                 kb_chunk_ratio: float = 0.6,
                 shared_chunk_ratio: float = 0.3,
                 unique_chunk_ratio: float = 0.1):
        
        assert abs(kb_chunk_ratio + shared_chunk_ratio + unique_chunk_ratio - 1.0) < 0.01
        
        self.num_kb_chunks = num_kb_chunks
        self.num_queries = num_queries
        self.chunks_per_query = chunks_per_query
        self.kb_chunk_ratio = kb_chunk_ratio
        self.shared_chunk_ratio = shared_chunk_ratio
        self.unique_chunk_ratio = unique_chunk_ratio
        
        # 系统提示词（所有查询共享）
        self.system_prompt = Chunk("SYSTEM_PROMPT")
        
        # 创建知识库chunks
        self.kb_chunks = [Chunk(f"KB_C{i}") for i in range(num_kb_chunks)]
        
        # 创建共享的新chunks（模拟热门内容）
        self.shared_new_chunks = [Chunk(f"SHARED_C{i}") for i in range(50)]
        
        self.unique_chunk_counter = 0
    
    def _select_chunks(self) -> List[Chunk]:
        """为一个查询选择chunks（不包括system prompt）"""
        num_chunks = random.randint(*self.chunks_per_query)
        chunks = []
        
        for _ in range(num_chunks):
            rand = random.random()
            if rand < self.kb_chunk_ratio:
                chunks.append(random.choice(self.kb_chunks))
            elif rand < self.kb_chunk_ratio + self.shared_chunk_ratio:
                chunks.append(random.choice(self.shared_new_chunks))
            else:
                unique_chunk = Chunk(f"UNIQUE_C{self.unique_chunk_counter}")
                self.unique_chunk_counter += 1
                chunks.append(unique_chunk)
        
        return chunks
    
    def generate_queries(self) -> List[Query]:
        """生成所有查询"""
        queries = []
        for i in range(self.num_queries):
            content_chunks = self._select_chunks()
            # 每个查询都包含system prompt
            all_chunks = [self.system_prompt] + content_chunks
            queries.append(Query(i, all_chunks))
        return queries
    
    def run_experiment(self) -> Dict:
        """运行完整实验"""
        print("Generating queries...")
        queries = self.generate_queries()
        
        print("Running Baseline (Standard Prefix Cache)...")
        baseline = BaselinePrefixCache()
        for query in queries:
            baseline.process_query(query)
        
        print("Running FusionRAG (With Alternative Path)...")
        fusionrag = FusionRAGCache()
        for query in queries:
            fusionrag.process_query(query)
        
        # 计算冗余计算次数（同一个chunk被计算多次）
        baseline_redundant = sum(max(0, count - 1) for count in baseline.chunk_recompute_count.values())
        fusionrag_redundant = sum(max(0, count - 1) for count in fusionrag.chunk_recompute_count.values())
        
        # 计算统计数据
        results = {
            'num_queries': self.num_queries,
            'kb_chunk_ratio': self.kb_chunk_ratio * 100,
            'shared_chunk_ratio': self.shared_chunk_ratio * 100,
            'unique_chunk_ratio': self.unique_chunk_ratio * 100,
            'chunks_per_query': f"{self.chunks_per_query[0]}-{self.chunks_per_query[1]}",
            
            'baseline': {
                'total_storage_gb': baseline.total_storage_mb / 1024,
                'total_computations': sum(baseline.chunk_recompute_count.values()),
                'redundant_computations': baseline_redundant,
                'unique_chunks': len(baseline.chunk_recompute_count),
                'total_chunk_accesses': baseline.total_chunk_accesses,
                'cache_hits': baseline.cache_hits,
                'cache_misses': baseline.cache_misses,
                'overall_hit_rate': baseline.cache_hits / baseline.total_chunk_accesses * 100 if baseline.total_chunk_accesses > 0 else 0,
            },
            
            'fusionrag': {
                'total_storage_gb': fusionrag.total_storage_mb / 1024,
                'total_computations': sum(fusionrag.chunk_recompute_count.values()),
                'redundant_computations': fusionrag_redundant,
                'unique_chunks': len(fusionrag.chunk_recompute_count),
                'alternative_path_hits': fusionrag.alternative_path_hits,
                'total_attempts': fusionrag.total_alternative_path_attempts,
                'alt_path_hit_rate': fusionrag.alternative_path_hits / fusionrag.total_alternative_path_attempts * 100 if fusionrag.total_alternative_path_attempts > 0 else 0,
                # 总体缓存命中率
                'total_chunk_accesses': fusionrag.total_chunk_accesses,
                'prefix_cache_hits': fusionrag.prefix_cache_hits,
                'alternative_path_cache_hits': fusionrag.alternative_path_cache_hits,
                'cache_misses': fusionrag.cache_misses,
                'total_cache_hits': fusionrag.prefix_cache_hits + fusionrag.alternative_path_cache_hits,
                'overall_hit_rate': (fusionrag.prefix_cache_hits + fusionrag.alternative_path_cache_hits) / fusionrag.total_chunk_accesses * 100 if fusionrag.total_chunk_accesses > 0 else 0,
            },
        }
        
        # 计算改进
        results['improvements'] = {
            'storage_reduction': (baseline.total_storage_mb - fusionrag.total_storage_mb) / baseline.total_storage_mb * 100 if baseline.total_storage_mb > 0 else 0,
            'redundant_computation_reduction': (baseline_redundant - fusionrag_redundant) / baseline_redundant * 100 if baseline_redundant > 0 else 0,
            'hit_rate_improvement': results['fusionrag']['overall_hit_rate'] - results['baseline']['overall_hit_rate'],
        }
        
        return results

def print_results(results: Dict):
    """打印实验结果"""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS: Alternative Path Storage Overhead Reduction")
    print("="*70)
    
    print(f"\nExperimental Setup:")
    print(f"  Total Queries: {results['num_queries']}")
    print(f"  Chunk Distribution:")
    print(f"    - Knowledge Base Chunks: {results['kb_chunk_ratio']:.0f}%")
    print(f"    - Shared New Chunks: {results['shared_chunk_ratio']:.0f}%")
    print(f"    - Unique Chunks: {results['unique_chunk_ratio']:.0f}%")
    print(f"  Chunks per Query: {results['chunks_per_query']}")
    
    print(f"\nBaseline (Standard Prefix Cache):")
    print(f"  Total Storage: {results['baseline']['total_storage_gb']:.2f} GB")
    print(f"  Total Computations: {results['baseline']['total_computations']}")
    print(f"  Redundant Computations: {results['baseline']['redundant_computations']}")
    print(f"  Unique Chunks: {results['baseline']['unique_chunks']}")
    print(f"  Total Chunk Accesses: {results['baseline']['total_chunk_accesses']}")
    print(f"  Cache Hits: {results['baseline']['cache_hits']}")
    print(f"  Cache Misses: {results['baseline']['cache_misses']}")
    print(f"  Overall Hit Rate: {results['baseline']['overall_hit_rate']:.1f}%")
    
    print(f"\nFusionRAG (With Alternative Path):")
    print(f"  Total Storage: {results['fusionrag']['total_storage_gb']:.2f} GB")
    print(f"  Total Computations: {results['fusionrag']['total_computations']}")
    print(f"  Redundant Computations: {results['fusionrag']['redundant_computations']}")
    print(f"  Unique Chunks: {results['fusionrag']['unique_chunks']}")
    print(f"  Total Chunk Accesses: {results['fusionrag']['total_chunk_accesses']}")
    print(f"  Prefix Cache Hits: {results['fusionrag']['prefix_cache_hits']}")
    print(f"  Alternative Path Cache Hits: {results['fusionrag']['alternative_path_cache_hits']}")
    print(f"  Total Cache Hits: {results['fusionrag']['total_cache_hits']}")
    print(f"  Cache Misses: {results['fusionrag']['cache_misses']}")
    print(f"  Overall Hit Rate: {results['fusionrag']['overall_hit_rate']:.1f}%")
    print(f"  Alternative Path Hit Rate (when attempted): {results['fusionrag']['alt_path_hit_rate']:.1f}%")
    
    print(f"\nImprovements:")
    print(f"  Storage Reduction: {results['improvements']['storage_reduction']:.1f}%")
    print(f"  Redundant Computation Reduction: {results['improvements']['redundant_computation_reduction']:.1f}%")
    print(f"  Hit Rate Improvement: {results['improvements']['hit_rate_improvement']:.1f}%")
    
    print("\n" + "="*70)
    
    # 生成可以直接填入论文的文本
    print("\n📝 For Paper (replace XX with these values):")
    print(f"\n  'we simulate {results['num_queries']} queries'")
    print(f"  'where {results['kb_chunk_ratio']:.0f}% of chunks already exist in the knowledge base,'")
    print(f"  '{results['shared_chunk_ratio']:.0f}% are shared among users but newly uploaded,'")
    print(f"  'and {results['unique_chunk_ratio']:.0f}% are completely unique.'")
    print(f"\n  'FusionRAG reduces total KVCache storage by {results['improvements']['storage_reduction']:.1f}%'")
    print(f"  'baseline recomputes the same chunk\\'s KVCache {results['baseline']['redundant_computations']} times'")
    print(f"  'FusionRAG eliminates {results['improvements']['redundant_computation_reduction']:.1f}% of redundant computations'")
    print(f"  'Alternative Path mechanism achieves hit rates of {results['fusionrag']['alt_path_hit_rate']:.1f}% when attempted'")
    print(f"  'FusionRAG achieves an overall cache hit rate of {results['fusionrag']['overall_hit_rate']:.1f}%'")
    print(f"  'improving upon baseline\\'s {results['baseline']['overall_hit_rate']:.1f}% by {results['improvements']['hit_rate_improvement']:.1f} percentage points'")

def test_example():
    """测试你给的例子：Query1: [S,1,5], Query2: [S,1,2,3,5]"""
    print("\n" + "="*70)
    print("Testing Your Example")
    print("="*70)
    
    # 创建chunks
    S = Chunk("S")
    C1 = Chunk("1")
    C2 = Chunk("2")
    C3 = Chunk("3")
    C5 = Chunk("5")
    
    # Query1: [S, 1, 5]
    query1 = Query(1, [S, C1, C5])
    
    # Query2: [S, 1, 2, 3, 5]
    query2 = Query(2, [S, C1, C2, C3, C5])
    
    print("\nQuery1: [S, 1, 5]")
    print("Query2: [S, 1, 2, 3, 5]")
    
    # Baseline
    print("\n--- Baseline ---")
    baseline = BaselinePrefixCache()
    result1 = baseline.process_query(query1)
    print(f"Query1: Computed chunks: {result1['chunks_computed']}")
    
    result2 = baseline.process_query(query2)
    print(f"Query2: Computed chunks: {result2['chunks_computed']}")
    print(f"Total storage: {baseline.total_storage_mb:.2f} MB")
    print(f"Overall hit rate: {baseline.cache_hits}/{baseline.total_chunk_accesses} = {baseline.cache_hits/baseline.total_chunk_accesses*100:.1f}%")
    
    # FusionRAG
    print("\n--- FusionRAG (With Alternative Path) ---")
    fusionrag = FusionRAGCache()
    result1 = fusionrag.process_query(query1)
    print(f"Query1: Computed chunks: {result1['chunks_computed']}, Alt path hits: {result1['alt_path_hits']}")
    
    result2 = fusionrag.process_query(query2)
    print(f"Query2: Computed chunks: {result2['chunks_computed']}, Alt path hits: {result2['alt_path_hits']}")
    print(f"Total storage: {fusionrag.total_storage_mb:.2f} MB")
    print(f"Prefix cache hits: {fusionrag.prefix_cache_hits}")
    print(f"Alternative path cache hits: {fusionrag.alternative_path_cache_hits}")
    print(f"Overall hit rate: {fusionrag.prefix_cache_hits + fusionrag.alternative_path_cache_hits}/{fusionrag.total_chunk_accesses} = {(fusionrag.prefix_cache_hits + fusionrag.alternative_path_cache_hits)/fusionrag.total_chunk_accesses*100:.1f}%")
    
    print("\n期望结果：")
    print("  Query2处理时，S和1应该通过prefix cache命中")
    print("  2和3需要重新计算（系统中没有）")
    print("  5应该通过alternative path命中（尝试[S,1,2,5]失败后，尝试[S,1,5]成功）")

if __name__ == "__main__":
    # 先测试你的例子
    test_example()
    
    # 运行完整实验
    print("\n\n")
    simulator = RAGSimulator(
        num_kb_chunks=100,
        num_queries=1000,
        chunks_per_query=(3, 8),
        kb_chunk_ratio=0.6,
        shared_chunk_ratio=0.3,
        unique_chunk_ratio=0.1
    )
    
    results = simulator.run_experiment()
    print_results(results)