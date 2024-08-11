

class CreateRefBankNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
        }}
    RETURN_TYPES = ("REF_BANK",)
    FUNCTION = "create"

    CATEGORY = "reference"

    def create(self):
        return ({},)
