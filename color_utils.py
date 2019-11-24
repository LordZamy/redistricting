from collections import defaultdict

def color_list_to_synthetic_id(color_list):
    """
    color_list: {(synthetic_id: string) : (color: int)} 
    returns {(color: int) : [synthetic_id: string]}
    """
    color_sythetic_id_map = defaultdict(list)
    for synthetic_id, color in color_list.items():
        color_sythetic_id_map[color].append(synthetic_id)
    
    return color_sythetic_id_map

def synth_map_to_matched_precincts(color_synthetic_id_map, matching_fn):
    """
    color_synthetic_id_map: {(color: int) : [synthetic_id: string]}
    matching_fn: Should return (matched_precinct_name, county_name) for a given
    synthetic id. Can return `None` if no match found.
    returns {(color: int) : [(matched_precinct_name, county_name)]}
    """
    color_list = defaultdict(list)
    for color, id_list in color_synthetic_id_map.items():
        for synthetic_id in id_list:
            # should get a tuple of (matched_precinct_name, county_name) or None
            match_tup = matching_fn(synthetic_id)
            if match_tup == None:
                continue
            color_list[color].append(match_tup)
    
    return color_list

def color_list_to_matched_precincts(color_list, matching_fn):
    color_synthetic_id_map = color_list_to_synthetic_id(color_list)
    return synth_map_to_matched_precincts(color_synthetic_id_map, matching_fn)