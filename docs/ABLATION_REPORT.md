# Ablation Study Report

**Date:** December 31, 2024  
**Test Queries:** 5 diverse questions  
**Components Tested:** Hierarchical Chunking, Graph Search

---

## Results Summary

### Hierarchical Chunking
- **Accuracy Impact:** 0% (same scores due to small doc)
- **Speed Impact:** +45% faster (0.60s → 0.33s)
- **Conclusion:** Positive for performance, needs larger docs for accuracy benefit

### Graph Search
- **Coverage:** 40% of queries (relationship type only)
- **Score Improvement:** 19x higher (0.664 → 12.8)
- **Success Rate:** 100% for valid relationship queries (2/2)
- **Conclusion:** Critical component for relationship questions

### Raw Data
See: `data/ablation_results.json`

---

## Recommendations

1. ✅ Keep hierarchical chunking (speed benefit)
2. ✅ Keep graph search (critical for relationships)
3. ⏳ Add keyword search for completeness
4. ⏳ Test with larger documents for better hierarchical results

---

END OF REPORT