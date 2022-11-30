from ast import literal_eval

# typed-entity 스페셜 토큰 추가
def add_entity_token(row, tem):
    """
    tem == 1 :〈Something〉는 #%PER%조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.
    tem == 2 :〈Something〉는 <e2><e4>PER</e4>조지 해리슨</e2>이 쓰고 <e1><e3>ORG</e3>비틀즈</e1>가 1969년 앨범 《Abbey Road》에 담은 노래다.
    """
    # entity token list. tem == 1 : 특수기호 토큰, tem == 2 : 스페셜 토큰
    etl = [[], ["@", "@", "#", "#", "*", "*", "%", "%"], ["<e1>", "</e1>", "<e2>", "</e2>", "<e3>", "</e3>", "<e4>", "</e4>"]]

    sent = row["sentence"]  # sentence
    se = literal_eval(row["subject_entity"])  # subject entity
    oe = literal_eval(row["object_entity"])  # object entity
    se["end_idx"] = se["start_idx"] + len(se["word"].split(",")[0]) - 1
    oe["end_idx"] = oe["start_idx"] + len(oe["word"].split(",")[0]) - 1
    new_sent = ""
    if se["start_idx"] < oe["start_idx"]:  # 문장에 subject -> object 순으로 등장
        new_sent = (
            sent[: se["start_idx"]]
            + etl[tem][0]
            + etl[tem][4]
            + se["type"]
            + etl[tem][5]
            + sent[se["start_idx"] : se["end_idx"] + 1]
            + etl[tem][1]
            + sent[se["end_idx"] + 1 : oe["start_idx"]]
            + etl[tem][2]
            + etl[tem][6]
            + oe["type"]
            + etl[tem][7]
            + sent[oe["start_idx"] : oe["end_idx"] + 1]
            + etl[tem][3]
            + sent[oe["end_idx"] + 1 :]
        )
    else:  # 문장에 object -> subject 순으로 등장
        new_sent = (
            sent[: oe["start_idx"]]
            + etl[tem][2]
            + etl[tem][6]
            + oe["type"]
            + etl[tem][7]
            + sent[oe["start_idx"] : oe["end_idx"] + 1]
            + etl[tem][3]
            + sent[oe["end_idx"] + 1 : se["start_idx"]]
            + etl[tem][0]
            + etl[tem][4]
            + se["type"]
            + etl[tem][5]
            + sent[se["start_idx"] : se["end_idx"] + 1]
            + etl[tem][1]
            + sent[se["end_idx"] + 1 :]
        )

    return new_sent
