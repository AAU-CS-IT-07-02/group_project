```mermaid
flowchart TD
    UPPAAL["UPPAAL (TA / SMC)"] -->|calls native C function| CLayer["C native function (libcurl)"]
    CLayer -->|HTTP POST /predict| PythonAPI["Python Inference API\n(FastAPI, preloaded model)"]
    PythonAPI -->|JSON response| CLayer
    CLayer -->|return value| UPPAAL
    subgraph LocalHost
        CLayer
        PythonAPI
    end
    note right of PythonAPI: Model preloaded at server start
```