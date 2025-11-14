# Benchmark Dashboard - Complete Explanation

This document explains how the benchmark dashboard displays metrics and calculates statistics.

---

## 📊 Graph Display (Efficiency, Cost, and Latency)

### Overview
The benchmark page displays a **bar chart** that compares three deployment platforms:
- **Serverless**
- **Traditional**
- **Hybrid**

### How the Graph Works

#### 1. **Data Source**
The graph gets its data from the `/benchmark-data` API endpoint, which returns:

```json
{
  "platforms": ["Serverless", "Traditional", "Hybrid"],
  "cost": [0.14, 0.12, 0.10],
  "latency_ms": [70, 90, 65],
  "efficiency_percent": [88, 82, 95]
}
```

#### 2. **Three Metrics Available**
The graph can display one of three metrics at a time:

| Metric | Description | Unit | Color | Data Array |
|--------|-------------|------|-------|------------|
| **Efficiency** | Resource utilization efficiency | Percentage (%) | Green (`rgba(34,197,94,0.9)`) | `efficiency_percent` |
| **Cost** | Deployment cost | US Dollars ($) | Blue (`rgba(96,165,250,0.9)`) | `cost` |
| **Latency** | Response time | Milliseconds (ms) | Orange (`rgba(249,115,22,0.9)`) | `latency_ms` |

#### 3. **Graph Rendering Process**

**Step 1: Data Extraction**
```javascript
// Extract arrays from backend response
let costArr = [0.14, 0.12, 0.10];      // Cost values for 3 platforms
let latArr = [70, 90, 65];              // Latency values for 3 platforms
let effArr = [88, 82, 95];              // Efficiency values for 3 platforms
```

**Step 2: Metric Selection**
- User clicks one of three buttons: **Efficiency**, **Cost**, or **Latency**
- The selected metric determines which array is displayed
- Default metric is **Efficiency**

**Step 3: Chart Creation**
- Uses **Chart.js** library to create a bar chart
- X-axis: Platform names (`["Serverless", "Traditional", "Hybrid"]`)
- Y-axis: Selected metric values (with appropriate unit)
- Each bar is colored according to the metric:
  - Efficiency → Green bars
  - Cost → Blue bars
  - Latency → Orange bars

**Step 4: Display Format**
- **Efficiency**: Shows as `88 %`, `82 %`, `95 %`
- **Cost**: Shows as `$0.14`, `$0.12`, `$0.10`
- **Latency**: Shows as `70 ms`, `90 ms`, `65 ms`

#### 4. **Visual Example**

When displaying **Cost**:
```
Y-axis: Cost ($)
  0.14 |     ████
  0.12 |  ████
  0.10 |████
       └─────────────────────
      Serverless Traditional Hybrid
```

When displaying **Efficiency**:
```
Y-axis: Efficiency (%)
    95 |              ████
    88 |  ████
    82 |     ████
       └─────────────────────
      Serverless Traditional Hybrid
```

---

## 📈 Summary Statistics Calculation

Below the graph, three summary statistics are displayed:

### 1. **Average Cost (Avg Cost)**

#### What it is:
The **mean (average)** of all cost values across all platforms.

#### Calculation:
```javascript
function avg(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Example with data: [0.14, 0.12, 0.10]
costAvg = (0.14 + 0.12 + 0.10) / 3
        = 0.36 / 3
        = 0.12
```

#### Formula:
```
Avg Cost = (Sum of all costs) / (Number of platforms)
         = (cost₁ + cost₂ + cost₃) / 3
```

#### Display:
- Rounded to 2 decimal places
- Example: `$0.12`

---

### 2. **P50 Latency (50th Percentile / Median)**

#### What it is:
The **median** latency value - 50% of values are below this, 50% are above.

#### Calculation:
```javascript
function percentile(values, p) {
  // Step 1: Sort the array in ascending order
  const arr = [...values].sort((a, b) => a - b);
  
  // Step 2: Calculate index position
  // For p50: index = (50/100) * (length - 1)
  const idx = Math.round((p / 100) * (arr.length - 1));
  
  // Step 3: Return value at that index
  return arr[idx];
}

// Example with data: [70, 90, 65]
// Step 1: Sort → [65, 70, 90]
// Step 2: Index = (50/100) * (3-1) = 0.5 * 2 = 1
// Step 3: Value at index 1 = 70
latP50 = 70 ms
```

#### Step-by-Step Example:
1. **Original data**: `[70, 90, 65]`
2. **Sort ascending**: `[65, 70, 90]`
3. **Calculate index**: 
   - For 50th percentile: `index = (50/100) × (3-1) = 1`
4. **Get value**: `arr[1] = 70`
5. **Result**: `70 ms`

#### Formula:
```
P50 = Value at position: (50/100) × (n-1)
where n = number of values
```

#### Display:
- Rounded to 0 decimal places (whole number)
- Example: `70 ms`

---

### 3. **P25 Latency (25th Percentile / First Quartile)**

#### What it is:
The **first quartile** - 25% of values are below this, 75% are above.

#### Calculation:
```javascript
// Same percentile function, but with p = 25
function percentile(values, 25) {
  const arr = [...values].sort((a, b) => a - b);
  const idx = Math.round((25 / 100) * (arr.length - 1));
  return arr[idx];
}

// Example with data: [70, 90, 65]
// Step 1: Sort → [65, 70, 90]
// Step 2: Index = (25/100) * (3-1) = 0.25 * 2 = 0.5 → rounds to 1
// Step 3: Value at index 1 = 70
latP25 = 70 ms
```

#### Step-by-Step Example:
1. **Original data**: `[70, 90, 65]`
2. **Sort ascending**: `[65, 70, 90]`
3. **Calculate index**: 
   - For 25th percentile: `index = (25/100) × (3-1) = 0.5` → rounds to `1`
4. **Get value**: `arr[1] = 70`
5. **Result**: `70 ms`

#### Formula:
```
P25 = Value at position: (25/100) × (n-1)
where n = number of values
```

#### Display:
- Rounded to 0 decimal places (whole number)
- Example: `70 ms`

---

## 🔍 Detailed Code Flow

### Complete Data Flow:

```
1. Backend (/benchmark-data endpoint)
   ↓
   Returns JSON with arrays:
   {
     "cost": [0.14, 0.12, 0.10],
     "latency_ms": [70, 90, 65],
     "efficiency_percent": [88, 82, 95]
   }
   ↓
2. Frontend (benchmark.js)
   ↓
   Extracts arrays:
   - costArr = [0.14, 0.12, 0.10]
   - latArr = [70, 90, 65]
   - effArr = [88, 82, 95]
   ↓
3. Graph Rendering
   ↓
   User selects metric → Chart.js renders bar chart
   - X-axis: ["Serverless", "Traditional", "Hybrid"]
   - Y-axis: Selected metric values
   ↓
4. Summary Statistics
   ↓
   Calculate:
   - Avg Cost = avg(costArr) = 0.12
   - P50 Latency = percentile(latArr, 50) = 70
   - P25 Latency = percentile(latArr, 25) = 70
   ↓
5. Display
   ↓
   Show badges below graph:
   - Avg Cost: $0.12
   - p50 Latency: 70 ms
   - p25 Latency: 70 ms
```

---

## 📝 Important Notes

### Current Implementation:
- **Mock Data**: The benchmark data is currently **mock/sample data** from the backend
- **Fixed Values**: The values are hardcoded in `backend/app.py` (lines 314-329)
- **Three Platforms**: Always compares exactly 3 platforms

### Percentile Calculation:
- Uses **nearest-rank method** (not interpolation)
- For small arrays (3 values), P25 and P50 may be the same
- The calculation rounds the index to the nearest integer

### Graph Behavior:
- **Responsive**: Chart adjusts to container size
- **Interactive**: Hover shows exact values
- **Toggleable**: Switch between Efficiency, Cost, and Latency
- **Color-coded**: Each metric has a distinct color

---

## 🎯 Real-World Example

### Sample Data:
```json
{
  "cost": [0.14, 0.12, 0.10],
  "latency_ms": [70, 90, 65],
  "efficiency_percent": [88, 82, 95]
}
```

### Calculations:

**Average Cost:**
```
(0.14 + 0.12 + 0.10) / 3 = 0.12
Display: $0.12
```

**P50 Latency:**
```
Sorted: [65, 70, 90]
Index: (50/100) × (3-1) = 1
Value: 70
Display: 70 ms
```

**P25 Latency:**
```
Sorted: [65, 70, 90]
Index: (25/100) × (3-1) = 0.5 → 1
Value: 70
Display: 70 ms
```

### Graph Display (when Cost is selected):
```
Cost ($)
  0.14 |     ████  ← Serverless: $0.14
  0.12 |  ████     ← Traditional: $0.12
  0.10 |████       ← Hybrid: $0.10
       └─────────────────────
      Serverless Traditional Hybrid
```

---

## 🔧 Code References

### Key Functions:

1. **`renderSimpleChart()`** (lines 48-101)
   - Creates the Chart.js bar chart
   - Handles color, formatting, and labels

2. **`percentile()`** (lines 40-45)
   - Calculates percentile values
   - Sorts array and finds value at calculated index

3. **`renderSummary()`** (lines 132-157)
   - Calculates and displays summary statistics
   - Creates badge elements for Avg Cost, P50, P25

4. **`avg()`** (line 136)
   - Simple average calculation
   - Sum divided by count

---

## 📚 Additional Resources

- **Chart.js Documentation**: https://www.chartjs.org/
- **Percentile Explanation**: https://en.wikipedia.org/wiki/Percentile
- **Statistical Measures**: Mean, Median, Quartiles

---

**Last Updated**: Based on current codebase implementation
**File Locations**:
- Frontend: `frontend/static/benchmark.js`
- Backend: `backend/app.py` (lines 307-330)

