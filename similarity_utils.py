# similarity_utils.py
import re
import pypinyin
import Levenshtein
from cnradical import Radical, RunOption

# 初始化 cnradical
radical = Radical(RunOption.Radical)

########################################
# 1. 拼音相似度
########################################
def split_pinyin_syllable(syllable: str):
    """
    拆分拼音音节为 (声母, 韵母, 声调)，适配零声母和两字母声母。
    例:
        - 'zhang3' -> ('zh', 'ang', '3')
        - 'ai4'    -> ('', 'ai', '4')  # 零声母
    """
    match_tone = re.search(r'(\d)$', syllable)
    tone = match_tone.group(1) if match_tone else ''
    core = re.sub(r'\d$', '', syllable)

    # 优先匹配两字母声母
    initials = [
        'zh', 'ch', 'sh',
        'b', 'p', 'm', 'f',
        'd', 't', 'n', 'l',
        'g', 'k', 'h',
        'j', 'q', 'x',
        'r', 'z', 'c', 's'
    ]

    initial = ''
    for ini in initials:
        if core.startswith(ini):
            initial = ini
            break

    final = core[len(initial):] if initial else core  # 零声母保留全部

    return initial, final, tone



def compute_pinyin_similarity(candidate: str, target: str) -> float:
    """
    升级版拼音相似度：
    1) 分别获取单字的 pinyin（可能有多音字，只取第一个）
    2) 将音节拆分为 (initial, final, tone)
    3) 计算各自的相似度后加权
       - 声母相等 -> +0.4
       - 韵母编辑距离 -> 根据编辑距离归一化
       - 声调完全相等 -> +0.1
    如果非中文或获取失败，降级为原先的编辑距离
    """

    # 若 candidate 或 target 不一定只是一字，这里假设只处理单个字符
    # 如果你的 fill-mask 可能生成多个字，这里需要额外处理
    cand_pinyin_list = pypinyin.pinyin(candidate, style=pypinyin.Style.TONE3, heteronym=False)
    targ_pinyin_list = pypinyin.pinyin(target, style=pypinyin.Style.TONE3, heteronym=False)

    if not cand_pinyin_list or not targ_pinyin_list:
        # fallback: 与原先一样，用完整拼音做编辑距离
        return fallback_pinyin_dist(candidate, target)

    cand_p = cand_pinyin_list[0][0]  # 取第一种拼音
    targ_p = targ_pinyin_list[0][0]

    # 拆分 (initial, final, tone)
    ci, cf, ct = split_pinyin_syllable(cand_p)
    ti, tf, tt = split_pinyin_syllable(targ_p)

    # 1) 比较声母
    score = 0.0
    if ci == ti:
        score += 0.4

    # 2) 比较韵母: 用编辑距离归一化
    max_len = max(len(cf), len(tf)) or 1
    dist = Levenshtein.distance(cf, tf)
    # 相似度：1 - (dist / max_len)
    final_sim = 1 - dist / max_len
    # 这部分再给 0.5 权重
    score += 0.5 * final_sim

    # 3) 比较声调
    if ct == tt and ct != '':
        score += 0.1

    # 限制到 [0,1]
    return min(score, 1.0)


def fallback_pinyin_dist(candidate, target):
    """若无法拆分音节，就走原先的编辑距离方案"""
    cand_pinyin = "".join(pypinyin.lazy_pinyin(candidate))
    targ_pinyin = "".join(pypinyin.lazy_pinyin(target))
    if cand_pinyin == targ_pinyin:
        return 1.0
    max_len = max(len(cand_pinyin), len(targ_pinyin))
    return 1 - Levenshtein.distance(cand_pinyin, targ_pinyin) / max_len


########################################
# 2. 字形相似度
########################################
def compute_shape_similarity(candidate: str, target: str) -> float:
    """
    基于笔画数+部首的简单字形相似度：
     - 如果两字完全相同 -> 1.0
     - 否则: 笔画相似度 + (部首相同小额奖励)
       shape_sim = 0.5 * strokeSim + 0.2 * radBonus
       并再乘个系数让整体不超过 1
    如果取不到笔画或部首，则退化到0.5
    """
    if candidate == target:
        return 1.0

    # 若不是单一中文字符，直接返回 0.5
    if len(candidate) != 1 or len(target) != 1:
        return 0.5

    # 获取笔画和部首
    try:
        radC, strokeC = radical.trans_ch(candidate)
        radT, strokeT = radical.trans_ch(target)

        # 如果获取失败（返回None或空），fallback
        if not strokeC or not strokeT:
            return 0.5

        strokeC = float(strokeC)
        strokeT = float(strokeT)
        # 计算笔画相似度
        strokeSim = 1 - abs(strokeC - strokeT) / max(strokeC, strokeT)

        # 部首奖励
        radBonus = 1.0 if radC == radT and radC else 0.0

        shape_sim = 0.5 * strokeSim + 0.2 * radBonus
        # 限制到 [0,1]，防止超出
        shape_sim = min(1.0, shape_sim)
        # 最低留个保底值
        return max(shape_sim, 0.0)

    except Exception:
        # 如果出错，按 0.5 返回
        return 0.5


########################################
# 3. 综合打分
########################################
def get_all_candidate_scores(candidate_list, target, alpha, beta, gamma):
    """
    综合得分 = alpha * 模型得分 + beta * 拼音相似度 + gamma * 字形相似度
    返回排序后的元组 (token_str, combined, model_score, pinyin, shape)
    """
    results = []
    for cand in candidate_list:
        token = cand["token_str"]
        model_score = cand["score"]

        pinyin_sim = compute_pinyin_similarity(token, target)
        shape_sim = compute_shape_similarity(token, target)

        combined = alpha * model_score + beta * pinyin_sim + gamma * shape_sim
        results.append((token, combined, model_score, pinyin_sim, shape_sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
