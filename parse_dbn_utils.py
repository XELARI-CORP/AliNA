import ast
import tqdm
import numpy as np          # used by get_consensus_dbn (not called in "skip" but needed for file completeness)
from typing import List, Set, Optional, Tuple
import naskit as nsk

### CHECK
# get_approved_coev_dbns + 
# get_raw_coev_dbns
# combine_true_and_coev_dbn + 
# get_consensus_dbn - 

def remove_gaps_from_dbn(native_seq, gapped_seq, gapped_dbn):

    if ('T' in native_seq) and ('U' in gapped_seq):
        gapped_seq = gapped_seq.replace('U','T')
        
    assert native_seq == gapped_seq.replace('N',''), "native and gapped w/o 'N' sequences do not match"
    assert len(gapped_dbn) == len(gapped_seq), "gapped seq and dbn must have the same length"

    native_dbn = ''.join(gapped_dbn[i] for i, c in enumerate(gapped_seq) if c != 'N')
    return native_dbn


def get_consensus_dbn(seq, dbns, f=0.1):
        
    if not dbns:
        return ""

    th = np.ceil(len(dbns)*f)
    
    seq_len = len(seq)
    offsets = [(-2,1),(-2,2),(-1,1),(-1,2),(1,-2),(1,-1),(2,-2),(2,-1)]
    
    # 1. Frequency Matrix
    freq = np.zeros((seq_len, seq_len))
    for dbn in dbns:
        adj = nsk.NA(seq, dbn).get_adjacency()
        freq += adj

    
    freq[freq<=th] = 0
    # 2. Find conflict-free pairs (row sum + col sum == 2*cell)
    row_sums = freq.sum(axis=1)
    col_sums = freq.sum(axis=0)
    is_conflict_free = (freq > 0) & ( freq == (row_sums[:,None] + col_sums[None,:])/2 )
    
    consensus_adj = np.triu(is_conflict_free, k=1).astype(float) # excluding main diagonal

    freq = np.triu(freq, k=1)
    freq[consensus_adj>0] = 0

    neib = np.full((seq_len, seq_len), -1)
    neib[freq>0] = 0
    
    rs, cs = np.where(consensus_adj>0)
    for r, c in zip(rs, cs):
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (0 <= nr < nc < seq_len) and (neib[nr,nc]>-1):
                neib[nr, nc] += 1

    while True:

        m = np.max(neib)
        if m == -1:
            break
            
        rs, cs = np.where(neib == m)

        best_freq = -1
        best_pair = None

        for r, c in zip(rs, cs):
            if freq[r,c] > best_freq:
                best_freq = freq[r,c]
                best_pair = (r,c)

        r, c = best_pair

        consensus_adj[r,c] = 1
        freq[r,:] = 0
        freq[c,:] = 0
        freq[:,r] = 0
        freq[:,c] = 0

        neib[r,:] = -1
        neib[c,:] = -1
        neib[:,r] = -1
        neib[:,c] = -1

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (0 <= nr < nc < seq_len) and (neib[nr,nc]>-1):
                neib[nr, nc] += 1

    final_adj = consensus_adj + consensus_adj.T
    return nsk.NA(seq).from_adjacency(final_adj).struct

# def quantize_matrix(M, threshold = 0.5):
#     seq_length = M.shape[-1]
#     diag = 1 - torch.diag(torch.ones(seq_length))
#     fm = torch.zeros((seq_length, seq_length), dtype=torch.int32)
    
#     thmask = M>threshold
#     s = M*diag*thmask

#     while torch.sum(s)>0:
#         m = int(s.argmax())
#         r = m//seq_length
#         c = m%seq_length

#         fm[r,c] = 1
#         s[r] = 0
#         s[c] = 0
#         s[:, r] = 0
#         s[:, c] = 0

#     x = (fm + fm.T)
#     return x

def get_approved_coev_dbns(na):
    
    coev_dbns = ast.literal_eval(na.meta['coev_dbns'])
    dbns_set = set()
    
    true_adj = na.get_adjacency() # target
    
    for coev_dbn in coev_dbns:
        coev_adj = nsk.NA(coev_dbn).get_adjacency()
        n_tp = (coev_adj*true_adj).sum().item()
        n_fp = (coev_adj*(1-true_adj)).sum().item()
        if (n_tp>0) and (n_fp>0) and (coev_dbn not in dbns_set):
            dbns_set.add(coev_dbn)
            
        elif (n_tp==0) and (n_fp>0):
            comb_dbn = combine_true_and_coev_dbn(na.struct, coev_dbn)
            if (comb_dbn is not None) and (comb_dbn not in dbns_set):
                dbns_set.add(comb_dbn)

    return list(dbns_set)

def get_raw_coev_dbns(na):
    
    coev_dbns = ast.literal_eval(na.meta['coev_dbns'])
    dbns_set = set()
    
    brackets = set("([{}])<>")
    
    for dbn in coev_dbns:
        if not any(p in brackets for p in dbn):
            continue
        if (dbn not in dbns_set):
            dbns_set.add(dbn)

    return list(dbns_set)

def combine_true_and_coev_dbn(true_dbn, coev_dbn):
    n_fp_th = 2
    l = len(coev_dbn)
        
    true_pairs = list(nsk.NA(true_dbn).pairs)
    coev_pairs = list(nsk.NA(coev_dbn).pairs)
    
    true_nts = {nt for pair in true_pairs for nt in pair}

    good_fp_pairs = []
    for pair in coev_pairs:
        if len(set(pair) & true_nts)==0:
            good_fp_pairs.append(pair)
            
    if len(good_fp_pairs)<n_fp_th:
        return None
    else:
        all_pairs = true_pairs + good_fp_pairs
    
        adj = np.zeros((l,l), dtype=np.int32)
        r, c = zip(*all_pairs)
        adj[r,c] = 1
        adj[c,r] = 1
    
        na = nsk.NucleicAcid.from_adjacency(adj)
        
        return na.struct

def align_structure_to_msa(seq: str, struct: str, aligned_seq: str, name: str):
    """
    Inserts gaps ('.') into a DBN structure to match an aligned sequence.
    
    Args:
        seq: The original sequence (no gaps).
        struct: The secondary structure (same length as seq).
        aligned_seq: The sequence with '-' representing gaps.
    """
    # Validation: target and structure must match
    assert len(seq) == len(struct), f"{name}: Structure and target sequence must have the same length."
    
    # Validation: aligned_seq without gaps must match seq
    assert aligned_seq.replace('-', '') == seq, f"{name}: Aligned sequence (minus '-') must match target sequence."

    aligned_struct = []
    struct_iter = iter(struct)
    
    for nt in aligned_seq:
        if nt == '-' or nt == 'N':
            # Insert a structure gap corresponding to the sequence gap
            aligned_struct.append('.')
        else:
            # Pull the next available structure character
            aligned_struct.append(next(struct_iter))

    assert len(aligned_struct) == len(aligned_seq), f"{name}: Aligned structure and sequence must have the same length."
    return "".join(aligned_struct)
### CORE FUNCTION ###

def deduplicate_keep_order_msa(msa: List[str]) -> List[str]:
    """Deduplicate sequences while preserving order."""
    query = msa[0]
    seen = {query}
    result = [query]
    for seq in msa[1:]:
        if seq not in seen:
            seen.add(seq)
            result.append(seq)
    return result


def create_nucleotide_mapper() -> dict:
    """Create translation table for nucleotide mapping."""
    map_from = 'TBKMRSVWY'
    map_to   = 'UNNNNNNNN'
    return str.maketrans(map_from, map_to)


def validate_sequence(seq: str, valid_nucleotides: Set[str]) -> bool:
    """Check if sequence contains only valid nucleotides."""
    return all(nt in valid_nucleotides for nt in seq)


def process_seq(na: nsk.NA, valid_nucleotides: Set[str], 
                mapper: dict) -> tuple:
    """
    Process MSA: align target, deduplicate, filter invalid sequences.
    Returns (filtered_msa, aligned_target_seq, skip_flag)
    """
    msa_data = [] # will be populated from meta if 'msa' key exists
    
    skip_msa = False
    msa_contains_target = False
    msa_and_target_have_equal_length = False
    
    aligned_target_seq = na.seq.upper().translate(mapper)  # presume it may contain '-'
    clean_target_seq = aligned_target_seq.replace("-", "") # default assumption

    if 'msa' in na.meta.keys():
        
        msa_data = ast.literal_eval(na.meta['msa'])
        if msa_data: # not empty
        # if empty then put target seq to msa
            msa_and_target_have_equal_length = len(aligned_target_seq) == len(msa_data[0])
            for seq in msa_data:
                clean_seq = seq.upper().translate(mapper).replace("-", "")
                if clean_seq == clean_target_seq:
                    msa_contains_target = True
                    aligned_target_seq = seq.upper().translate(mapper)
                    break
                    
        skip_msa = (not msa_data) or ( \
                    (not msa_contains_target) and \
                    (not msa_and_target_have_equal_length) \
                    )

    if skip_msa:
        msa_data = [aligned_target_seq.replace('-', 'N')]  # Clear out the un-mergeable MSA data
    else:
        msa_data = [aligned_target_seq] + msa_data
        msa_data = [seq.upper().translate(mapper).replace('-', 'N') for seq in msa_data]
        msa_data = [seq for seq in msa_data if validate_sequence(seq, valid_nucleotides)]
        msa_data = deduplicate_keep_order_msa(msa_data)

    return msa_data, aligned_target_seq, skip_msa


def process_struct(na, dbns_policy: str,
                   aligned_target_seq: str,
                   target_seq: str,
                   target_struct_key: str) -> tuple:
    """
    Process coevolutionary structures and target structure.
    """
    assert dbns_policy in ['approve', 'raw', 'skip'], f"invalid dbns_policy value: {dbns_policy}"
    assert (dbns_policy != 'skip') or (target_struct_key != 'coev_dbns'), \
        f"target_struct_key == 'coev_dbns', but dbns_policy == 'skip'"
    # Default values
    skip = False
    msg = ""
    brackets = set("<([{}])>")
    
    # 1. Get coevolutionary structures
    coev_dbns_aligned, coev_dbns = [], []
    if dbns_policy == 'approve':
        coev_dbns = get_approved_coev_dbns(na)
    elif dbns_policy == 'raw':
        coev_dbns = get_raw_coev_dbns(na)
    
    if (dbns_policy != 'skip'):
        if not coev_dbns:
            skip = True
            msg = "empty coev_dbns, but dbns_policy != 'skip'"
        else:
            for dbn in coev_dbns:
                dbn_aligned = align_structure_to_msa(target_seq, dbn, aligned_target_seq, na.name) #!
                coev_dbns_aligned.append(dbn_aligned)

    # 2. Get target structure
    target_struct_aligned = '.'*len(aligned_target_seq) # dummy placeholder
    
    if target_struct_key == "coev_dbns":
        ...
        # target_struct_aligned = get_consensus_dbn(aligned_target_seq, coev_dbns_aligned) # SHOULD BE VERIFIED!
        assert False # debug
    elif target_struct_key in na.meta.keys():
        target_struct_aligned = align_structure_to_msa(
            target_seq, na.meta[target_struct_key], aligned_target_seq, na.name)
        
    elif target_struct_key == "default":
        target_struct_aligned = align_structure_to_msa(
            target_seq, na.struct, aligned_target_seq, na.name)
    else:
        assert False, f"invalid target_struct_key value: {target_struct_key}"
        
    if not any(p in brackets for p in target_struct_aligned):
        # skip = True
        msg = "empty target_struct_aligned" if not msg else (msg + " and empty target_struct_aligned")
    
    return coev_dbns_aligned, target_struct_aligned, skip, msg


def setup_logging(log_file: Optional[str]):
    """Setup logging handle."""
    log_handle = open(log_file, "w") if log_file else None
    
    def log_skip(reason: str, index: int, name: str):
        msg = f"Skip #{index} ({name}): {reason}\n"
        if log_handle:
            log_handle.write(msg)
        else:
            print(msg)
    
    return log_handle, log_skip

def prepare_nas_for_alina(path: str,
                          names: Optional[List[str]] = None):
    # Summary. Read RNA sequences from a file and filter them:
    # (len<=256) & (2nd structure, e.g. there're base pairs) & (only AUGC bases, e.g. RNA)
    # return: list of naskit.NA objects
    
    if names is not None:
        names_set = set(names)
        print(f"#names={len(names_set)}")
    else:
        names_set = None
        print("#names=None (reading all structures)")
    
    # 1. Read and initial filter by name
    nas = []
    with nsk.dotRead(path) as f:
        for na in f:
            if (na is not None):
                if names_set is None or na.name in names:
                    nas.append(na)
                    
    print(f"Before: {len(nas)}", end="; ")
    
    # Define valid bases as a set outside the loop for O(1) lookups
    VALID_BASES = set("AUGCT")
    
    # 2. Comprehensive filtering
    # Optimized the character check by ensuring every character is a subset of VALID_BASES
    nas = [
        na for na in nas 
        if len(na) <= 256 
        and len(na.pairs) != 0 
        and set(na.seq).issubset(VALID_BASES)
    ]
    
    print(f"after: {len(nas)}")
    return nas


def prepare_nas_for_elina(
                path: str,
                target_struct_key: str,
                names: List[str] = None,
                log_file: Optional[str] = None,
                meta_keys: Optional[List[str]] = None, 
                dbns_policy: str = 'skip') -> List[nsk.NA]:
    """
    Parses a secondary structure file format (.dbn, .nsk) via dotRead, filters and maps sequences 
    by name match, cleanses/validates nucleotide representations, processes MSAs, 
    aligns consensus structures, and instantiates processed NA structures.

    Summary
    -------
    This function acts as a processing pipeline that reads biological sequence data from 
    a file path. It filters entries to keep only those present in the `names` whitelist. 
    Valid sequences undergo nucleotide normalization (via mapping tables), target 
    Multiple Sequence Alignment (MSA) parsing with fallback validation, structural policy 
    application (handling downstream dot-bracket annotations), and optional metadata inclusion. 
    It returns a filtered list of fully prepared `nsk.NA` computational objects.

    Arguments
    ---------
    path : str
        The file path pointing to the input structural record file (e.g., dotRead compatible format).
    names : List[str]
        A whitelist subset of sequence identifier strings. Only records whose `na.name` 
        property matches an element in this list will be processed.
    target_struct_key : Optional[str], default=None
        The key identifier matching specific target consensus structural formats within the 
        file's metadata parameters.
    log_file : Optional[str], default=None
        File path to write processing exception logs and sequence skipping indicators. If 
        None, logging handles are bypassed.
    meta_keys : Optional[List[str]], default=None
        Explicit list of metadata keys to extract from the source record and preserve within 
        the final object's metadata dictionary.
    dbns_policy : str, default='skip'
        Policy constraint configuration governing how structural discrepancies or dot-bracket 
        strings are handled. Accepted variants include 'approve', 'raw', or 'skip'.

    Returns
    -------
    List[nsk.NA]
        A list of initialized, validated, and normalized nucleic acid structure tracking 
        objects conforming to downstream sequence alignment specifications.
    """
    # Initialize
    mapper = create_nucleotide_mapper()
    valid_nucleotides = set("AGTCUN")
    names_set = set(names) if names is not None else None
    sample_size = len(names) if names is not None else None
    
    nas = []
    n, skipped = 0, 0
    
    # Setup logging
    log_handle, log_skip = setup_logging(log_file)
    
    try:
        with nsk.dotRead(str(path), raise_na_errors=False, 
                         upper_sequence=False, meta_separator=": ") as f:
            
            for j, na in tqdm.tqdm(enumerate(f)):
                try:
                    name = na.name
                    if (names_set is not None) and (name not in names_set):
                        continue
                    
                    # Process sequence
                    target_seq = na.seq.upper().translate(mapper)
                    if not validate_sequence(target_seq, valid_nucleotides):
                        log_skip("target seq contains uncommon nt", j, name)
                        skipped += 1
                        continue
                    
                    # Process target MSA
                    msa, aligned_target_seq, skip_msa = process_seq(
                        na, valid_nucleotides, mapper
                    )
                    
                    if skip_msa:
                        log_skip(f"warning: msa processing failed: only target seq used", j, name)
                    
                    # Process structures
                    aligned_coev_dbns, aligned_target_struct, skip_struct, msg = process_struct(
                        na, dbns_policy, aligned_target_seq, 
                        target_seq, target_struct_key
                    )
                    
                    if skip_struct:
                        log_skip(f"warning: structure processing failed: {msg}", j, name)
                        #skipped += 1
                        # continue
                    
                    # Build metadata
                    meta = {
                        "original_seq": na.seq,
                        "msa": msa, 
                        "coev_dbns": aligned_coev_dbns
                    }
                    
                    if meta_keys is not None:
                        for k in meta_keys:
                            if k in na.meta:
                                meta[k] = na.meta[k]
                    
                    # Create final NA object
                    aligned_target_seq = aligned_target_seq.replace('-', 'N')
                    na_obj = nsk.NA(
                        aligned_target_seq,
                        aligned_target_struct,
                        name=name,
                        meta=meta
                    )
                    
                    nas.append(na_obj)
                    n += 1
                    
                except Exception as e:
                    log_skip(f"{type(e).__name__}: {e}", j, name if 'name' in locals() else 'unknown')
                    skipped += 1
                    
    finally:
        if log_handle:
            log_handle.close()
            
    if sample_size is None:
        sample_size = n + skipped
        
    print(f"passed  : {n}/{sample_size}")
    print(f"skipped : {skipped}/{sample_size}")
    print(f"#nas    : {len(nas)}")
    return nas