#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include "cJSON.h"
#include <math.h>

/*
    uppaal_wrapper.c

    Purpose
    -------
    Minimal C helper functions for calling a local HTTP inference server (FastAPI)
    from UPPAAL via a native C interface. UPPAAL models can call the exported
    functions below by linking against the produced shared library (e.g.
    `libuppaal_nn.so`).

    Design notes and safety
    -----------------------
    - UPPAAL cannot directly receive array or heap-allocated pointer return values
        from external C functions in a convenient, model-safe way. For this reason
        this file provides two complementary patterns:
            1) `uppaal_nn_infer_scalar_fixed(...)` — compatibility helper that performs
                 an HTTP POST and returns a single scalar value for the requested
                 `room_id`.
            2) `uppaal_nn_update(...)` + `uppaal_nn_get_pred(int)` — recommended
                 pattern: call `uppaal_nn_update` once to perform the HTTP request and
                 cache the first prediction row in an internal static buffer; then call
                 `uppaal_nn_get_pred(i)` repeatedly to read each room's scalar value.

    - The static buffer (`latest_prediction`) is NOT thread-safe. If your
        environment calls these functions from multiple threads or reentrant
        contexts, you must add synchronization.

    - Keep the inference server bound to `127.0.0.1` and use fixed ports to avoid
        non-determinism in model checking (SMC). The C code uses a short timeout
        (3 seconds) to keep external calls bounded.

    - Exported functions return simple scalar types (`double` / `int`) so they
        map cleanly to UPPAAL's external function semantics.

    Ownership and memory
    --------------------
    - JSON payload builders return a heap-allocated string; callers must `free()`
        the returned pointer.
    - `WriteMemoryCallback` grows a caller-provided `MemoryStruct` buffer and
        ensures it's NUL-terminated. The caller is responsible for `free()`ing the
        `.memory` pointer when finished.

*/

struct MemoryStruct { char *memory; size_t size; };

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
        /*
            WriteMemoryCallback
            -------------------
            libcurl write callback that appends the incoming `contents` chunk into
            the caller-provided `MemoryStruct` buffer. The callback resizes the
            buffer with `realloc` so it can be used to collect arbitrarily-sized
            responses.

            Important:
            - `mem->memory` should be initialized by the caller (typically to
                `malloc(1)` and `mem->size = 0`) before the request.
            - On `realloc` failure we return 0 which tells libcurl to abort the
                transfer; callers should check for incomplete responses.
            - The buffer is NUL-terminated so it can be passed to JSON parsers.
        */
        char *ptr = realloc(mem->memory, mem->size + realsize + 1);
        if(ptr == NULL) return 0; /* OOM - abort transfer */
        mem->memory = ptr;
        memcpy(&(mem->memory[mem->size]), contents, realsize);
        mem->size += realsize;
        mem->memory[mem->size] = 0; /* NUL-terminate for parsers */
        return realsize;
}

/*
Build payload from arrays; return allocated string (caller free)

Parameters:
 - y0: initial output vector (length y0_len)
 - controls_flat: flattened controls buffer (length controls_len). When
   `nest_controls` is non-zero the function will wrap the entire flattened
   vector as a single row (useful when the HTTP API expects an array-of-rows
   where each row is itself an array). When `nest_controls` is zero the
   flattened buffer is emitted directly as a top-level JSON array.
 - method: integration method string (e.g. "rk4") or NULL
 - is_normalized: 0/1 flag indicating whether inputs are already normalized

Returns: pointer to an allocated JSON string (caller must free), or NULL on
error.
*/
static char *build_payload_from_flat(const double *y0, int y0_len,
                                    const double *controls_flat, int controls_len,
                                    int nest_controls, const char *method, int is_normalized) {
    cJSON *root = cJSON_CreateObject();
    if(!root) return NULL;
    cJSON *y0obj = cJSON_CreateDoubleArray(y0, y0_len);
    if(!y0obj) { cJSON_Delete(root); return NULL; }
    cJSON_AddItemToObject(root, "y0", y0obj);

    if(nest_controls) {
        cJSON *controls = cJSON_CreateArray();
        if(!controls) { cJSON_Delete(root); return NULL; }
        cJSON *row = cJSON_CreateDoubleArray(controls_flat, controls_len);
        if(!row) { cJSON_Delete(root); return NULL; }
        cJSON_AddItemToArray(controls, row);
        cJSON_AddItemToObject(root, "controls", controls);
    } else {
        cJSON *controls = cJSON_CreateDoubleArray(controls_flat, controls_len);
        if(!controls) { cJSON_Delete(root); return NULL; }
        cJSON_AddItemToObject(root, "controls", controls);
    }

    if(method) cJSON_AddStringToObject(root, "method", method);
    cJSON_AddNumberToObject(root, "is_normalized", is_normalized ? 1 : 0);

    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return s;
}

// UPPAAL-friendly wrapper: 6 y0 values + 36 flattened control doubles
//
// Convenience function kept for backwards compatibility. It performs a
// synchronous HTTP POST to the server and returns a single scalar value for
// the requested `room_id` from the first predicted output row.
//
// Notes:
//  - `room_id` must be in the range [0, NUM_ROOMS-1] for sensible results.
//  - The call is blocking and has a short timeout (3 seconds set via libcurl).
//  - If the HTTP request fails, the JSON payload is malformed, or the requested
//    index is not present, the function returns `NAN`.
//
// This function is useful when UPPAAL only needs a single room value per call.
// If you need all room temperatures at once, prefer calling
// `uppaal_nn_update(...)` followed by repeated `uppaal_nn_get_pred(i)` calls.
double uppaal_nn_infer_scalar_fixed(int room_id,
    double y0_0,double y0_1,double y0_2,double y0_3,double y0_4,double y0_5,
    double c0,double c1,double c2,double c3,double c4,double c5,double c6,double c7,double c8,double c9,
    double c10,double c11,double c12,double c13,double c14,double c15,double c16,double c17,double c18,double c19,
    double c20,double c21,double c22,double c23,double c24,double c25,double c26,double c27,double c28,double c29,
    double c30,double c31,double c32,double c33,double c34,double c35
) {
    const char *url = "http://127.0.0.1:8000/infer";
    double y0[6] = {y0_0,y0_1,y0_2,y0_3,y0_4,y0_5};
    double controls[36] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,
                           c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,
                           c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,
                           c30,c31,c32,c33,c34,c35};

    char *payload = build_payload_from_flat(y0, 6, controls, 36, 1, "rk4", 0);
    if(!payload) return NAN;

    struct MemoryStruct resp;
    resp.memory = malloc(1);
    resp.size = 0;

    CURL *curl = curl_easy_init();
    if(!curl) { free(payload); free(resp.memory); return NAN; }

    struct curl_slist *hdrs = NULL;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);

    /* Perform single update of internal prediction buffer, then return requested room */
    CURLcode rc = curl_easy_perform(curl);

    double out = NAN;
    if(rc == CURLE_OK) {
        cJSON *j = cJSON_Parse(resp.memory);
        if(j) {
            cJSON *pred = cJSON_GetObjectItemCaseSensitive(j, "prediction");
            if(cJSON_IsArray(pred)) {
                cJSON *first_row = cJSON_GetArrayItem(pred, 0);
                if(first_row && cJSON_IsArray(first_row)) {
                    cJSON *first_val = cJSON_GetArrayItem(first_row, room_id);
                    if(first_val && cJSON_IsNumber(first_val)) out = first_val->valuedouble;
                }
            }
            cJSON_Delete(j);
        }
    }

    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    free(payload);
    free(resp.memory);
    return out;
}

/*
    Alternative approach (recommended)
    -------------------------------
    The functions below implement a two-step pattern that is safer and more
    efficient when the UPPAAL model needs multiple room values at once:

    1) `uppaal_nn_update(...)` performs the HTTP POST to `/infer`, parses the
         response and stores the first output row into the internal static buffer
         `latest_prediction`.

    2) `uppaal_nn_get_pred(int room_id)` simply returns the stored scalar value
         for `room_id` from that buffer.

    Benefits:
    - Only a single blocking network call is performed for all room values.
    - UPPAAL sees only scalar-returning functions, which maps cleanly to its
        external function model.

    Important implementation details and caveats:
    - `latest_prediction` is a static, process-global buffer. It is NOT
        thread-safe or reentrancy-safe. If multiple contexts call `uppaal_nn_update`
        concurrently you must add synchronization.
    - The functions return simple status values (int success) or `double`
        scalars. UPPAAL models should check return codes where appropriate.
    - The C code uses libcurl with a 3-second timeout to keep calls bounded for
        model checking. Adjust as needed but keep deterministic behavior for SMC.
*/

#define NUM_ROOMS 6
static double latest_prediction[NUM_ROOMS] = {NAN, NAN, NAN, NAN, NAN, NAN};

int uppaal_nn_update(
    double y0_0,double y0_1,double y0_2,double y0_3,double y0_4,double y0_5,
    double c0,double c1,double c2,double c3,double c4,double c5,double c6,double c7,double c8,double c9,
    double c10,double c11,double c12,double c13,double c14,double c15,double c16,double c17,double c18,double c19,
    double c20,double c21,double c22,double c23,double c24,double c25,double c26,double c27,double c28,double c29,
    double c30,double c31,double c32,double c33,double c34,double c35
) {
    const char *url = "http://127.0.0.1:8000/infer";
    double y0[6] = {y0_0,y0_1,y0_2,y0_3,y0_4,y0_5};
    double controls[36] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,
                           c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,
                           c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,
                           c30,c31,c32,c33,c34,c35};

     /* Build the JSON body for the POST. Caller-provided payload must be
         freed by this function after use; build_payload_from_flat returns an
         owned heap pointer. */
     char *payload = build_payload_from_flat(y0, 6, controls, 36, 1, "rk4", 0);
     if(!payload) return 0;

    struct MemoryStruct resp;
    resp.memory = malloc(1);
    resp.size = 0;

    CURL *curl = curl_easy_init();
    if(!curl) { free(payload); free(resp.memory); return 0; }

    struct curl_slist *hdrs = NULL;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);

    /* Perform the HTTP request synchronously. On success parse the JSON and
       copy the first output row into the `latest_prediction` static buffer.
       We tolerate absent values by setting `NAN` for missing entries. */
    CURLcode rc = curl_easy_perform(curl);
    int success = 0;
    if(rc == CURLE_OK) {
        cJSON *j = cJSON_Parse(resp.memory);
        if(j) {
            cJSON *pred = cJSON_GetObjectItemCaseSensitive(j, "prediction");
            if(cJSON_IsArray(pred)) {
                cJSON *first_row = cJSON_GetArrayItem(pred, 0);
                if(first_row && cJSON_IsArray(first_row)) {
                    for(int i = 0; i < NUM_ROOMS; ++i) {
                        cJSON *val = cJSON_GetArrayItem(first_row, i);
                        if(val && cJSON_IsNumber(val)) latest_prediction[i] = val->valuedouble;
                        else latest_prediction[i] = NAN; /* keep semantics explicit */
                    }
                    success = 1;
                }
            }
            cJSON_Delete(j);
        }
    }

    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    free(payload);
    free(resp.memory);
    return success;
}

double uppaal_nn_get_pred(int room_id) {
    /* Return the stored prediction for `room_id`.
       Note: this is a simple accessor into the static buffer and does not
       perform any network I/O. If `uppaal_nn_update` has not been called or
       if the last update failed, the buffer entries may be `NAN`. */
    if(room_id < 0 || room_id >= NUM_ROOMS) return NAN;
    return latest_prediction[room_id];
}
