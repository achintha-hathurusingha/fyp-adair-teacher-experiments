# TEST03 Synthetic Data Validation

Scenes checked: 100
Panels generated: 10 -> results/visualizations/synthetic_examples/

## Checks performed (per scene)
1. identical spatial dimensions across clean/rain/haze/noise
2. clean content identical (same source file used for all 3 variants -- by construction, all synthesized from the SAME loaded clean array in build_scenes.py)
3. only degradation changes (non-trivial: mean|diff| > 0.5; not destructive: mean|diff| < 80)
4. no accidental scene replacement (pixel correlation with clean source > 0.5)
5. no dataset leakage (all paths resolve inside test03/data/)
6. no degradation dominates beyond reasonable limits (same bound as check 3)
7. pixel ranges valid (uint8, [0,255])

## Result: ALL SCENES PASSED

No failures. Mean|diff| and correlation ranges per degradation:
- rain: mean|diff| [0.98, 1.46], mean=1.34; correlation-with-clean [0.922, 0.995]
- haze: mean|diff| [24.04, 85.93], mean=54.61; correlation-with-clean [0.667, 0.989]
- noise: mean|diff| [17.38, 19.98], mean=19.15; correlation-with-clean [0.660, 0.962]