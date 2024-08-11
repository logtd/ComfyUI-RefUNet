
SD1_OUTPUT_MAP = set([])
for idx in [0,1,2,3,4,5,6,7,8,9]: # TODO check these
    SD1_OUTPUT_MAP.add(('output', idx))

SD1_MIDDLE_MAP = set([('middle', 0)])

SD1_INPUT_MAP = set()
for idx in [0,1,2,3,4,5,6,7,8,9,10,11]:
    SD1_INPUT_MAP.add(('input', idx))


SD1_REF_MAP = SD1_INPUT_MAP | SD1_MIDDLE_MAP | SD1_OUTPUT_MAP