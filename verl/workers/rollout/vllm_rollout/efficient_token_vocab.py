from transformers import AutoTokenizer

# Transition phrases
transition_phrases = [
    "think differently",
    "another way",
    "another approach",
    "another method",
    "another solution",
    "another strategy",
    "another technique"
]

# Reflection phrases
reflection_phrases = [
    "verify",
    "make sure",
    "hold on",
    "think again",
    "'s correct",
    "'s incorrect",
    "Let me check",
    "seems right"
]
slow_first_phrases = [
    "Okay",
    "Alright"
]



# Combine both lists if needed
slow_phrases = transition_phrases + reflection_phrases + slow_first_phrases


fast_first_phrases = ["To", "I", "First"]

fast_phrases =  fast_first_phrases

slow_phrases = [s.lower() for s in slow_phrases]
fast_phrases = [s.lower() for s in fast_phrases]


def detect_tokens(tokenizer, id_list):
    last_check_text=[]
    last_check_text.append(tokenizer.decode(id_list[-1:]))
    last_check_text.append(tokenizer.decode(id_list[-2:]))
    last_check_text.append(tokenizer.decode(id_list))
    
    for t in last_check_text:
        if t.lower() in slow_phrases:
            return -1
        if t.lower() in fast_phrases:
            return 1
    return 0

    


    



    
    


