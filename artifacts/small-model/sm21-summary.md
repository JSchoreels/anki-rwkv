# SM21 carry vs control (464 users, w8+s8)

dec12: carry curve 0.30213 (+1.50% vs fork), instant 0.27620 (+1.48%); zero curve 0.30214 (+1.51%), instant 0.27640 (+1.56%); A-B curve -0.00001 [-0.00030, +0.00034], instant -0.00020 -> FAIL: carry does not meet the rule

e12: carry curve 0.30416 (+2.18% vs fork), instant 0.27736 (+1.91%); zero curve 0.30471 (+2.37%), instant 0.27778 (+2.06%); A-B curve -0.00056 [-0.00110, +0.00001], instant -0.00041 -> FAIL: carry does not meet the rule

Rule: A(carry) - B(zero): curve <= -0.001 with upper95 < 0 AND instant <= +0.0005, 464 users, review-weighted, w8+s8
