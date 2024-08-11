

class CustomRefMapSD1Node:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
        }}
        for i in range(6):
            base['required'][f'input_{i}'] = ("BOOLEAN", { "default": True})

        base['required'][f'middle_0'] = ("BOOLEAN", { "default": True })

        for i in range(9):
            base['required'][f'output_{i}'] = ("BOOLEAN", { "default": True })
        
        return base
    RETURN_TYPES = ("ATTN_MAP",)
    FUNCTION = "apply"

    CATEGORY = "reference/custom"

    def apply(self, **kwargs):

        attention_map = set()
        for key, value in kwargs.items():
            if value:
                block, idx = key.split('_')
                attention_map.add((block, int(idx)))
            
        return (attention_map, )


class ConfigRefMapAdvNode:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
            "input_attns": ("STRING", {"multiline": True, "default": "0,1,2,3,4,5",  }),
            "middle_attns": ("STRING", {"multiline": True, "default": "0",  }),
            "output_attns": ("STRING", {"multiline": True, "default": "0,1,2,3,4,5,6,7,8" }),
        }}
        return base
    RETURN_TYPES = ("ATTN_MAP",)
    FUNCTION = "apply"

    CATEGORY = "reference/custom"

    def apply(self, input_attns, middle_attns, output_attns):

        attention_map = set()
        if input_attns != '' and input_attns is not None:
            for idx in input_attns.split(','):
                idx = idx.strip()
                if idx is '':
                    continue
                attention_map.add(('input', int(idx)))

        if middle_attns != '' and middle_attns is not None:
            for idx in middle_attns.split(','):
                idx = idx.strip()
                if idx is '':
                    continue
                attention_map.add(('middle', int(idx)))

        if output_attns != '' and output_attns is not None:
            for idx in output_attns.split(','):
                idx = idx.strip()
                if idx is '':
                    continue
                attention_map.add(('output', int(idx)))

        return (attention_map, )