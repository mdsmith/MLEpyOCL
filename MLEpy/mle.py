SITES = 1024
CHARACTERS = 64
NODES = 20

def naiveCPU():
    
    node_cache = [1.0/CHARACTERS] * CHARACTERS*SITES
    parent_cache = [1.0] * CHARACTERS*SITES
    model = [1.0/CHARACTERS] * CHARACTERS*CHARACTERS

    for n in range(NODES):
        for s in range(SITES):
            siteIndex = s*CHARACTERS
            for p in range(CHARACTERS):
                sum = 0.0
                parentIndex = p*CHARACTERS
                for c in range(CHARACTERS):
                    sum += node_cache[siteIndex + c] * model[parentIndex + c]
                parent_cache[s*CHARACTERS + p] *= sum


    print parent_cache[CHARACTERS*SITES-1]

if __name__ == "__main__":
    naiveCPU()
