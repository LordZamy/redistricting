from collections import defaultdict

def count_votes(vote_map, color_list):
    """
    Computes the votes per candidate for each district

    vote_map: {county: {precinct: {candidate: num_votes}}}
    color_list: {color: [(matched_precinct_name, county_name)]}

    returns color_votes: {color: {candidate: num_votes}}
    """
    color_votes = defaultdict(lambda: defaultdict(int))
    for color, precinct_county_list in color_list.items()
        for matched_precinct_name, county_name in precinct_county_list:
            for candidate, num_votes in vote_map[county_name][matched_precinct_name].items():
                color_votes[color][candidate] += num_votes

    return colors_votes

def winning_candidates(colors_votes):
    """
    colors_votes: {color: {candidate: num_votes}}

    returns colors_win: {color: candidate}
    """
    colors_wins = {}
    for color, candidate_vote_map in colors_votes.items():
        candidate, votes = max(candidate_vote_map.items(), key=lambda tup: tup[1])
        colors_wins[color] = candidate
    
    return color_wins

def winning_parties(color_wins, candidate_party_map):
    return {color: candidate_party_map[candidate] for color, candidate in color_wins.items()}

def get_winning_parties(vote_map, color_list, candidate_party_map):
    colors_votes = count_votes(vote_map, color_list)
    color_wins = winning_candidates(color_votes)
    return winning_parties(color_wins, candidate_party_map)
