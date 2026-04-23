import cv2
import numpy as np


LETTERS  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS   = "0123456789"
OPTIONS  = "ABCDE"

NAME_SHIFTS = dict(left=1, right=10, top=-30, bottom=5)

NUM_ANSWER_GROUPS    = 4
ROWS_PER_GROUP       = 10
OPTIONS_PER_QUESTION = 5

HOUGH_DP       = 1.2
HOUGH_MIN_DIST = 14
HOUGH_PARAM1   = 50
HOUGH_PARAM2   = 18
HOUGH_MIN_R    = 8
HOUGH_MAX_R    = 18

CLUSTER_GAP = 18 

COLOR_FILLED = (0, 200,   0)
COLOR_MULTI  = (0,   0, 255)
COLOR_EMPTY  = (0, 140, 255)
COLOR_NORMAL = (180,180,180)


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


def _top_two_scores(scores):
    s = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return s[0][0], s[0][1], s[1][1]


def _split_into_columns(region, num_cols):
    cw = region.shape[1] // num_cols
    return [region[:, i * cw:(i + 1) * cw] for i in range(num_cols)]


def _split_into_rows(region, num_rows):
    rh = region.shape[0] // num_rows
    return [region[i * rh:(i + 1) * rh, :] for i in range(num_rows)]


def _locate_region(image, template):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc


def _cluster_centers(values, gap=CLUSTER_GAP):
    arr = np.array(sorted(set(int(v) for v in values)))
    centers, cur = [], [arr[0]]
    for v in arr[1:]:
        if v - cur[-1] > gap:
            centers.append(int(np.median(cur)))
            cur = [v]
        else:
            cur.append(v)
    centers.append(int(np.median(cur)))
    return centers


def _detect_circles(gray):
    cs = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    return np.round(cs[0]).astype(int) if cs is not None else None


def _bubble_darkness(gray, bx, by, br, margin=2):
    patch = gray[
        max(0, by - br + margin): by + br - margin,
        max(0, bx - br + margin): bx + br - margin,
    ]
    return float(255 - np.mean(patch)) if patch.size > 0 else 0.0


def _find_adaptive_threshold(all_scores):
    arr = np.array(sorted(all_scores))
    gaps = [
        (arr[i + 1] - arr[i], float((arr[i] + arr[i + 1]) / 2))
        for i in range(len(arr) - 1)
    ]
    best_gap, best_mid = max(gaps, key=lambda x: x[0])
    return best_mid if best_gap > 10 else 50.0


def _classify_bubble(scores, threshold):
    filled = [i for i, s in enumerate(scores) if s >= threshold]
    if len(filled) == 0:
        return "EMPTY", None
    if len(filled) == 1:
        return "OK", OPTIONS[filled[0]]
    return "MULTIPLE", [OPTIONS[i] for i in filled]



def auto_straighten(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return image
    angles = [(l[0][1] * 180 / np.pi) - 90 for l in lines[:50]]
    M = cv2.getRotationMatrix2D(
        (image.shape[1] // 2, image.shape[0] // 2), np.median(angles), 1
    )
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def detect_bubble_grid(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th   = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [
        cv2.boundingRect(c) for c in contours
        if 80  < cv2.contourArea(c)     < 1200
        and 8  < cv2.boundingRect(c)[2] < 40
        and 8  < cv2.boundingRect(c)[3] < 40
    ]
    if len(bubbles) < 50:
        return None
    xs   = [b[0] for b in bubbles]
    left = [b for b in bubbles if np.percentile(xs, 3) <= b[0] <= np.percentile(xs, 82)]
    if len(left) < 50:
        return None
    lxs = [b[0] for b in left];  lys = [b[1] for b in left]
    x1  = int(np.percentile(lxs, 0))
    x2  = int(np.percentile([b[0] + b[2] for b in left], 96))
    y1  = sorted(lys)[int(0.075 * len(lys))]
    y2  = int(np.percentile([b[1] + b[3] for b in left], 99))
    if debug:
        dbg = image.copy(); cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        # cv2.imshow("Bubble Grid", dbg)
    return x1, y1, x2, y2


def extract_name_area(image, debug=False):
    grid = detect_bubble_grid(image, debug)
    if grid is None:
        return None
    x1, y1, x2, y2 = grid
    h, w = image.shape[:2]
    x1 = _clamp(x1 + NAME_SHIFTS["left"],   0, w)
    x2 = _clamp(x2 + NAME_SHIFTS["right"],  0, w)
    y1 = _clamp(y1 + NAME_SHIFTS["top"],    0, h)
    y2 = _clamp(y2 + NAME_SHIFTS["bottom"], 0, h)
    if debug:
        dbg = image.copy(); cv2.rectangle(dbg,(x1,y1),(x2,y2),(255,0,0),2)
        # cv2.imshow("Name Area", dbg)
    return image[y1:y2, x1:x2]


def detect_letters(name_area, num_cols=20, debug=False):
    result = []
    for col in _split_into_columns(name_area, num_cols):
        gray     = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        _, th    = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        rows     = _split_into_rows(th, 26)
        scores   = [cv2.countNonZero(r) / r.size for r in rows]
        best_idx, top1, top2 = _top_two_scores(scores)
        result.append(LETTERS[best_idx] if top1 >= 0.06 and (top1 - top2) > 0.05 else " ")
    return "".join(result)


def extract_center_number_area(image, name_area, debug=False):
    h_img, w_img   = image.shape[:2]
    h_name, w_name = name_area.shape[:2]
    nx, ny         = _locate_region(image, name_area)
    x1 = _clamp(nx + w_name + int(0.05  * w_img), 0, w_img)
    x2 = _clamp(x1           + int(0.195 * w_img), 0, w_img)
    y1 = _clamp(ny            + int(0.05  * h_name), 0, h_img)
    y2 = _clamp(y1            + int(0.40  * h_name), 0, h_img)
    if debug:
        dbg = image.copy(); cv2.rectangle(dbg,(x1,y1),(x2,y2),(255,0,0),2)
        # cv2.imshow("Centre Number Area", dbg)
    return image[y1:y2, x1:x2]


def detect_center_digits(center_area, num_cols=5, debug=False):
    result = []
    for col in _split_into_columns(center_area, num_cols):
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        th   = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        scores = []
        for row in _split_into_rows(th, 10):
            hc, wc = row.shape
            inner  = row[int(0.2*hc):int(0.8*hc), int(0.2*wc):int(0.8*wc)]
            scores.append(cv2.countNonZero(inner) / inner.size)
        best_idx, top1, top2 = _top_two_scores(scores)
        result.append(DIGITS[best_idx] if top1 >= 0.10 and (top1 - top2) > 0.04 else "_")
    return "".join(result)



def extract_answer_area(image, debug=False):
    h, w = image.shape[:2]
    y1 = int(0.643 * h);  y2 = int(0.885 * h)
    x1 = int(0.02  * w);  x2 = int(0.98  * w)
    if debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,200,0), 2)
        # cv2.imshow("Answer Area", cv2.resize(dbg, (800,900)))
    return image[y1:y2, x1:x2]


def detect_all_answers(answer_area, debug=False):

    aw      = answer_area.shape[1]
    gw      = aw // NUM_ANSWER_GROUPS

    all_scores   = []
    group_cache  = []

    for g in range(NUM_ANSWER_GROUPS):
        grp  = answer_area[:, g * gw:(g + 1) * gw]
        gray = cv2.cvtColor(grp, cv2.COLOR_BGR2GRAY)
        cs   = _detect_circles(gray)
        if cs is None:
            group_cache.append(None)
            continue

        col_c = sorted(_cluster_centers([int(c[0]) for c in cs]))[-OPTIONS_PER_QUESTION:]
        row_c = sorted(_cluster_centers([int(c[1]) for c in cs]))[:ROWS_PER_GROUP]

        rows = []
        for ri, ry in enumerate(row_c):
            row_cs = [c for c in cs if abs(int(c[1]) - ry) <= CLUSTER_GAP]
            scores, bcs = [], []
            for cx in col_c:
                bc = min(row_cs, key=lambda c: abs(int(c[0]) - cx)) if row_cs else None
                d  = _bubble_darkness(gray, int(bc[0]), int(bc[1]), int(bc[2])) if bc is not None else 0.0
                scores.append(d)
                bcs.append(bc)
                all_scores.append(d)
            rows.append((ri, ry, scores, bcs))

        group_cache.append((g, gray, col_c, rows))

    threshold = _find_adaptive_threshold(all_scores)

    results = {}
    for item in group_cache:
        if item is None:
            continue
        g, gray, col_c, rows = item
        for ri, ry, scores, bcs in rows:
            q              = g * ROWS_PER_GROUP + ri + 1
            status, answer = _classify_bubble(scores, threshold)
            results[q]     = {
                "status":    status,
                "answer":    answer,
                "scores":    scores,
                "threshold": threshold,
            }

    if debug:
        _draw_answer_debug(answer_area, group_cache, results, gw)

    return results, threshold


def _draw_answer_debug(answer_area, group_cache, results, gw):
    """Overlay coloured circles and labels on the answer area (debug mode)."""
    dbg = answer_area.copy()

    for item in group_cache:
        if item is None:
            continue
        g, gray, col_c, rows = item
        gx = g * gw

        for ri, ry, scores, bcs in rows:
            q      = g * ROWS_PER_GROUP + ri + 1
            info   = results[q]
            thr    = info["threshold"]
            status = info["status"]
            filled = [i for i, s in enumerate(scores) if s >= thr]

            for ci, bc in enumerate(bcs):
                if bc is None:
                    continue
                bx, by, br = int(bc[0]), int(bc[1]), int(bc[2])
                is_filled  = scores[ci] >= thr

                if status == "MULTIPLE" and is_filled:
                    color, thick = COLOR_MULTI,  2
                elif status == "OK" and is_filled:
                    color, thick = COLOR_FILLED, 2
                elif status == "EMPTY":
                    color, thick = COLOR_EMPTY,  1
                else:
                    color, thick = COLOR_NORMAL, 1

                cv2.circle(dbg, (gx + bx, by), br + 2, color, thick)

            # label anomalous rows
            lx, ly = gx + 4, int(ry)
            if status == "EMPTY":
                cv2.putText(dbg, f"Q{q}:EMPTY", (lx, ly - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_EMPTY, 1)
            elif status == "MULTIPLE":
                opts = "&".join(OPTIONS[i] for i in filled)
                cv2.putText(dbg, f"Q{q}:{opts}", (lx, ly - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_MULTI, 1)

    # cv2.imshow("Answer Debug", dbg)



def extract_dob_area(image, center_area, debug=False):
    h_img, w_img = image.shape[:2]
    h_c, w_c = center_area.shape[:2]

    cx, cy = _locate_region(image, center_area)

    left_shift  = -20   # move left (-) or right (+)
    right_shift = 45    # expand right
    x1 = cx + left_shift
    x2 = cx + w_c + right_shift

    y1 = cy + int(1.45 * h_c)
    y2 = y1 + int(0.98 * h_c)

    x1 = _clamp(x1, 0, w_img)
    x2 = _clamp(x2, 0, w_img)
    y1 = _clamp(y1, 0, h_img)
    y2 = _clamp(y2, 0, h_img)

    dob_area = image[y1:y2, x1:x2]

    if debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # cv2.imshow("DOB Area", dbg)
        # cv2.imshow("DOB Crop", dob_area)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return dob_area


def process_sheet(path, debug=False):
    debug = False
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read '{path}'")

    img = auto_straighten(auto_straighten(image))

    name_area = extract_name_area(img, debug=debug)
    name      = detect_letters(name_area, num_cols=20) if name_area is not None else ""

    centre = ""
    if name_area is not None:
        centre_area = extract_center_number_area(img, name_area, debug=debug)
        if centre_area is not None:
            centre = detect_center_digits(centre_area, num_cols=5)

    answer_area             = extract_answer_area(img, debug=debug)
    answers, threshold      = detect_all_answers(answer_area, debug=debug)
    
    dob = extract_dob_area(img, centre_area, debug=True)
    
    issues = {
        "empty":    [q for q, v in answers.items() if v["status"] == "EMPTY"],
        "multiple": {q: v["answer"] for q, v in answers.items() if v["status"] == "MULTIPLE"},
    }

    return {
        "name":          name.strip(),
        "centre_number": centre,
        "answers":       {q: v["answer"] for q, v in answers.items() if v["status"] == "OK"},
        "threshold":     threshold,
        "issues":        issues,
        "raw":           answers,
    }


# def main():
#     files = ["1.jpg", "2.jpg", "3.jpg"]

#     DEBUG_FILE = "1.jpg"   # 👈 change this anytime

#     for path in files:
#         debug = (path == DEBUG_FILE)   # 👈 only one file debug

#         try:
#             result = process_sheet(path, debug=debug)

#             print(f"\nFile: {path}")
#             print(f"Name: {result['name']}")
#             print(f"Centre: {result['centre_number']}")
#             print(f"Threshold: {result['threshold']:.1f}")

#             answers_formatted = []
#             raw = result["raw"]

#             for q in range(1, 41):
#                 v = raw.get(q, {})
#                 status = v.get("status")
#                 answer = v.get("answer")

#                 if status == "OK":
#                     val = answer
#                 elif status == "MULTIPLE":
#                     val = "&".join(answer)   # ['A','B'] → A&B
#                 elif status == "EMPTY":
#                     val = "·"
#                 else:
#                     val = "?"

#                 answers_formatted.append(f"Q {q}:{val}")

#             print(answers_formatted)

#         except FileNotFoundError as e:
#             print(f"Skipping: {e}")

# if __name__ == "__main__":
#     main()

def process_omr_file(file_path):
    try:
        result = process_sheet(file_path)

        formatted_answers = {}

        for q in range(1, 41):
            v = result["raw"].get(q, {})
            status = v.get("status")
            answer = v.get("answer")

            if status == "OK":
                formatted_answers[str(q)] = answer
            elif status == "MULTIPLE":
                formatted_answers[str(q)] = "&".join(answer)
            elif status == "EMPTY":
                formatted_answers[str(q)] = None
            else:
                formatted_answers[str(q)] = None

        return {
            "name": result["name"],
            "centre_number": result["centre_number"],
            "dob": None,
            "level": None,
            "answers": formatted_answers
        }

    except FileNotFoundError as e:
        return {
            "error": str(e)
        }
