LA = ('linear', 'adam')
LB = ('linear', 'lbfgs')
MA = ('monarch', 'adam')
MB = ('monarch', 'lbfgs')

N_EPOCHS_PER_PHASE = 250
N_TOTAL_MAX_EPOCHS = 1000
N_MAX_PHASES_FOR_PHASE_MONOID = N_TOTAL_MAX_EPOCHS // N_EPOCHS_PER_PHASE

def get_target_phase_sequences() -> set[tuple[tuple[str, str], ...]]:
    TARGET_PHASE_SEQUENCES = set()
    
    blocks_l_list_phases = [
        [LA], [LB],
        [LA, LA], [LB, LB], 
        [LA, LB]            
    ]
    blocks_m_list_phases = [
        [MA], [MB],
        [MA, MA], [MB, MB],
        [MA, MB]
    ]

    valid_blocks_l_phases = [tuple(b) for b in blocks_l_list_phases if 0 < len(b) <= N_MAX_PHASES_FOR_PHASE_MONOID]
    valid_blocks_m_phases = [tuple(b) for b in blocks_m_list_phases if 0 < len(b) <= N_MAX_PHASES_FOR_PHASE_MONOID]

    for blk_l in valid_blocks_l_phases:
        TARGET_PHASE_SEQUENCES.add(blk_l)

    for blk_m in valid_blocks_m_phases:
        TARGET_PHASE_SEQUENCES.add(blk_m)

    for blk_l in valid_blocks_l_phases:
        for blk_m in valid_blocks_m_phases:
            seq = list(blk_l) + list(blk_m) 
            if 0 < len(seq) <= N_MAX_PHASES_FOR_PHASE_MONOID:
                TARGET_PHASE_SEQUENCES.add(tuple(seq))

    for blk_m in valid_blocks_m_phases:
        for blk_l in valid_blocks_l_phases:
            seq = list(blk_m) + list(blk_l)
            if 0 < len(seq) <= N_MAX_PHASES_FOR_PHASE_MONOID:
                TARGET_PHASE_SEQUENCES.add(tuple(seq))
                
    return TARGET_PHASE_SEQUENCES

PRECOMPUTED_TARGET_PHASE_SEQUENCES = get_target_phase_sequences()

def monoid_phases(phase_word: list[tuple[str, str]]) -> list[tuple[str, str]]:
    current_phase_len = len(phase_word)
    if current_phase_len >= N_MAX_PHASES_FOR_PHASE_MONOID:
        return [] 

    possible_next_phase_types = set()
    current_phase_word_tuple = tuple(phase_word) 

    if not phase_word: 
        for target_phase_seq in PRECOMPUTED_TARGET_PHASE_SEQUENCES:
            if target_phase_seq: 
                possible_next_phase_types.add(target_phase_seq[0])
    else: 
        for target_phase_seq in PRECOMPUTED_TARGET_PHASE_SEQUENCES:
            if len(target_phase_seq) > current_phase_len and \
               target_phase_seq[:current_phase_len] == current_phase_word_tuple:
                possible_next_phase_types.add(target_phase_seq[current_phase_len]) 
    
    return sorted(list(possible_next_phase_types))

def convert_epochs_to_phases(epoch_word: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not epoch_word:
        return []
    
    num_epochs = len(epoch_word)
    
    if num_epochs % N_EPOCHS_PER_PHASE != 0:
        raise ValueError(
            f"La longueur du mot d'époques ({num_epochs}) doit être un multiple de "
            f"{N_EPOCHS_PER_PHASE} pour la conversion en phases."
        )

    phase_sequence = []
    for i in range(0, num_epochs, N_EPOCHS_PER_PHASE):
        phase_type = epoch_word[i] 
        phase_sequence.append(phase_type)
            
    return phase_sequence

def monoid_etendue(epoch_word: list[tuple[str, str]]) -> list[tuple[str, str]]:
    current_epoch_count = len(epoch_word)

    if current_epoch_count == N_TOTAL_MAX_EPOCHS:
        return [] 

    if current_epoch_count > 0 and current_epoch_count % N_EPOCHS_PER_PHASE != 0:
        return [epoch_word[-1]] 
    else: 
        phase_séquence_actuelle = convert_epochs_to_phases(epoch_word)
        return monoid_phases(phase_séquence_actuelle)

def _generate_words_recursive_helper(
    current_phase_sequence_list: list[tuple[str, str]], 
    all_found_phase_sequences: set[tuple[str,str]]
):
    current_phase_sequence_tuple = tuple(current_phase_sequence_list)

    if current_phase_sequence_list and current_phase_sequence_tuple in PRECOMPUTED_TARGET_PHASE_SEQUENCES:
        all_found_phase_sequences.add(current_phase_sequence_tuple)

    possible_next_phase_types = monoid_phases(current_phase_sequence_list)

    if not possible_next_phase_types: 
        return

    for phase_type in possible_next_phase_types:
        current_phase_sequence_list.append(phase_type) 
        _generate_words_recursive_helper(current_phase_sequence_list, all_found_phase_sequences)
        current_phase_sequence_list.pop()         

def get_all_possible_phase_sequences() -> list[list[tuple[str, str]]]:
    found_phase_sequences_set = set()
    _generate_words_recursive_helper([], found_phase_sequences_set)
    
    sorted_list_of_sequences = sorted(
        [list(seq_tuple) for seq_tuple in found_phase_sequences_set], 
        key=lambda s: (len(s), s) 
    )
    
    return sorted_list_of_sequences

if __name__ == '__main__':
    print(f"monoid_etendue([]) -> {monoid_etendue([])}")
    
    epoch_word_la_100 = [LA] * 100
    print(f"monoid_etendue(LA^100) -> {monoid_etendue(epoch_word_la_100)}")

    epoch_word_la_250 = [LA] * 250
    print(f"monoid_etendue(LA^250) -> {monoid_etendue(epoch_word_la_250)}")
    
    epoch_word_la250_lb10 = [LA] * 250 + [LB] * 10
    print(f"monoid_etendue(LA^250 + LB^10) -> {monoid_etendue(epoch_word_la250_lb10)}")

    epoch_word_la250_lb250 = [LA] * 250 + [LB] * 250
    print(f"monoid_etendue(LA^250 + LB^250) -> {monoid_etendue(epoch_word_la250_lb250)}")

    epoch_word_la_1000 = [LA] * 1000
    print(f"monoid_etendue(LA^1000) -> {monoid_etendue(epoch_word_la_1000)}")

    all_valid_phase_sequences = get_all_possible_phase_sequences()
    print(f"Nombre total de séquences de phases valides trouvées : {len(all_valid_phase_sequences)}")
    
    if len(all_valid_phase_sequences) > 10:
        print("Quelques exemples de séquences de phases valides (début et fin de la liste triée):")
        for i in list(range(min(5, len(all_valid_phase_sequences)))):
            print(f"  {all_valid_phase_sequences[i]}")
        if len(all_valid_phase_sequences) > 5: print("  ...")
        for i in list(range(max(5, len(all_valid_phase_sequences) - 5), len(all_valid_phase_sequences))):
             print(f"  {all_valid_phase_sequences[i]}")
    elif all_valid_phase_sequences:
        print("Séquences de phases valides:")
        for seq in all_valid_phase_sequences:
            print(f"  {seq}")