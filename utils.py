def build_dict():
    d = {}
    with open("tags.txt") as f:
        for line in f:
            split_list = line.split(' ', 1)
            key = split_list[0]
            val = split_list[1].strip()
            d[int(key)] = val
    return d
