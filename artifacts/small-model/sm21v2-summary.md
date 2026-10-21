# SM21 v2 lanes vs control and v1 (464 users, w8+s8)

dec12: lanes curve 0.30124 (+1.20% vs fork), instant 0.27514 (+1.10%); B (zero) curve 0.30214 (+1.51%), instant 0.27640 (+1.56%); C - B curve -0.00090 [-0.00136, -0.00023], instant -0.00125 -> FAIL

dec12: lanes curve 0.30124 (+1.20% vs fork), instant 0.27514 (+1.10%); A (v1 carry) curve 0.30213 (+1.50%), instant 0.27620 (+1.48%); C - A curve -0.00090 [-0.00139, -0.00033], instant -0.00105 -> FAIL (informational)

e12: lanes curve 0.30327 (+1.89% vs fork), instant 0.27639 (+1.55%); B (zero) curve 0.30471 (+2.37%), instant 0.27778 (+2.06%); C - B curve -0.00144 [-0.00206, -0.00071], instant -0.00138 -> PASS

e12: lanes curve 0.30327 (+1.89% vs fork), instant 0.27639 (+1.55%); A (v1 carry) curve 0.30416 (+2.18%), instant 0.27736 (+1.91%); C - A curve -0.00089 [-0.00139, -0.00036], instant -0.00097 -> FAIL (informational)

Rule: C(lanes) - B(zero): curve <= -0.001 with upper95 < 0 AND instant <= +0.0005, 464 users, review-weighted, w8+s8
