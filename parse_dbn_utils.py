import ast
import tqdm
import numpy as np          # used by get_consensus_dbn (not called in "skip" but needed for file completeness)
from typing import List, Set, Optional, Tuple
import naskit as nsk

### CHECK
# get_approved_coev_dbns
# get_raw_coev_dbns
# combine_true_and_coev_dbn
# get_consensus_dbn
# quantize_matrix

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

def align_structure_to_msa(target_seq: str, dbn_struct: str, aligned_seq: str, name: str):
    """
    Inserts gaps ('.') into a DBN structure to match an aligned sequence.
    
    Args:
        target_seq: The original sequence (no gaps).
        dbn_struct: The secondary structure (same length as target_seq).
        aligned_seq: The sequence with '-' representing gaps.
    """
    # Validation: target and structure must match
    assert len(target_seq) == len(dbn_struct), f"{name}: Structure and target sequence must have the same length."
    
    # Validation: aligned_seq without gaps must match target_seq
    assert aligned_seq.replace('-', '') == target_seq, f"{name}: Aligned sequence (minus '-') must match target sequence."

    new_struct = []
    struct_iter = iter(dbn_struct)
    
    for nt in aligned_seq:
        if nt == '-':
            # Insert a structure gap corresponding to the sequence gap
            new_struct.append('.')
        else:
            # Pull the next available structure character
            new_struct.append(next(struct_iter))
            
    return "".join(new_struct)
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
    
    skip = False
    msg = ""
    msa_data = [] # will be populated from meta if 'msa' key exists
    msa_contains_target = False
    
    aligned_target_seq = na.seq.upper().translate(mapper) # presume it may contain '-'
    clean_target_seq = aligned_target_seq.replace("-", "") # default assumption

    if 'msa' in na.meta.keys():
        
        msa_data = ast.literal_eval(na.meta['msa'])
        if msa_data: # not empty
        # if empty then put target seq to msa
            for seq in msa_data:
                clean_seq = seq.upper().translate(mapper).replace("-", "")
                if clean_seq == clean_target_seq:
                    msa_contains_target = True
                    aligned_target_seq = seq.upper().translate(mapper)
                    break
            skip = (not msa_contains_target) and (len(aligned_target_seq) != len(msa_data[0]))
    
    if not skip:
        msa_data = [aligned_target_seq] + msa_data
        msa_data = [seq.upper().translate(mapper).replace('-', 'N') for seq in msa_data]
        msa_data = [seq for seq in msa_data if validate_sequence(seq, valid_nucleotides)]
        msa_data = deduplicate_keep_order_msa(msa_data)

    return msa_data, aligned_target_seq, skip


def process_struct(na, dbns_policy: str,
                   aligned_target_seq: str,
                   target_seq: str,
                   target_struct_key: Optional[str]) -> tuple:
    """
    Process coevolutionary structures and target structure.
    Returns (coev_dbns_aligned, target_struct_aligned, skip_flag)
    """
    # default return values
    skip = False
    msg = ""
    target_struct_aligned = '.' # dummy placeholder
    coev_dbns_aligned = []
    
    brackets = set("([{}])<>")
    
    # Get coevolutionary structures
    if dbns_policy == 'approve':
        coev_dbns = get_approved_coev_dbns(na)
    elif dbns_policy == 'raw':
        coev_dbns = get_raw_coev_dbns(na)
    elif dbns_policy == "skip":
        coev_dbns = []
    else:
        coev_dbns = []
    
    if not coev_dbns and dbns_policy != 'skip':
        skip = True
        msg = "empty coev_dbns, but dbns_policy != 'skip'"
    
    # Align coevolutionary structures
    coev_dbns_aligned = []
    if dbns_policy != 'skip':
        for dbn in coev_dbns:
            dbn_aligned = align_structure_to_msa(target_seq, dbn, aligned_target_seq)
            coev_dbns_aligned.append(dbn_aligned)
    
    # Get target structure
    if target_struct_key is not None:
        if target_struct_key == "coev_dbns":
            if dbns_policy != 'skip':
                target_struct_aligned = get_consensus_dbn(aligned_target_seq, coev_dbns_aligned)
            else:
                skip = True
                msg = "target_struct_key == 'coev_dbns', but dbns_policy != 'skip'"
                target_struct_aligned = '.'
        elif target_struct_key in na.meta.keys():
            target_struct_aligned = align_structure_to_msa(
                target_seq, na.meta[target_struct_key], aligned_target_seq, na.name
            )
        elif target_struct_key == "default":
            target_struct_aligned = align_structure_to_msa(
                target_seq, na.struct, aligned_target_seq, na.name
            )
        else:
            skip = True
            msg = "unknown target_struct_key value"
        
        if not any(p in brackets for p in target_struct_aligned):
            skip = True
            msg = "empty target_struct_aligned"
    else:
        target_struct_aligned = '.' * len(aligned_target_seq)
    
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


def prepare_nas(path: str, names: List[str], target_struct_key: Optional[str] = None,
                log_file: Optional[str] = None, meta_keys: Optional[List[str]] = None, 
                sample_size: Optional[int] = None,
                dbns_policy: str = 'approve') -> List[nsk.NA]:
    """
    Prepare nucleic acid structures from a dotRead file.
    
    dbns_policy: 'approve', 'raw', or 'skip'
    """
    # Initialize
    mapper = create_nucleotide_mapper()
    valid_nucleotides = set("AGTCUN")
    names_set = set(names)
    sample_size = sample_size if sample_size is not None else len(names)
    
    nas = []
    n, skipped = 0, 0
    
    # Setup logging
    log_handle, log_skip = setup_logging(log_file)
    
    try:
        with nsk.dotRead(str(path), raise_na_errors=False, 
                         upper_sequence=False, meta_separator=": ") as f:
            
            for j, na in tqdm.tqdm(enumerate(f)):
                try:
                    if n >= sample_size:
                        break
                    
                    name = na.name
                    if name not in names_set:
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
                        log_skip(f"msa processing failed", j, name)
                        skipped += 1
                        continue
                    
                    # Process structures
                    aligned_coev_dbns, aligned_target_struct, skip_struct, msg = process_struct(
                        na, dbns_policy, aligned_target_seq, 
                        target_seq, target_struct_key
                    )
                    
                    if skip_struct:
                        log_skip(f"structure processing failed: {msg}", j, name)
                        skipped += 1
                        continue
                    
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
    
    print(f"passed  : {n}/{len(names)}")
    print(f"#skipped: {skipped}")
    print(f"#nas    : {len(nas)}")
    return nas