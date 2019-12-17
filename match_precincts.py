from matching import Player
from matching.games import StableMarriage
from fuzzywuzzy import process
import json
from collections import defaultdict

class PrecinctMatcher():
    def __init__(self, ga_precinct_map, election_precinct_map):
        """
        ga_precinct_map: dict from county_name -> list of precincts
        election_precinct_map: dict from coutny_name -> list of precincts
        
        This function returns a mapping from the GA shapefile precinct names (cleaned)
        to election precinct names.
        """
        assert len(ga_precinct_map) == len(election_precinct_map)
        self.ga_precinct_map = ga_precinct_map
        self.election_precinct_map = election_precinct_map
        self.effective_dummies = 0
        self.ga_player_map = {}
        self.perform_matching()

    def perform_matching(self):
        self.matching = {}
        for county in self.ga_precinct_map.keys():
            solved_matching = self.match_precincts_in_county(self.ga_precinct_map[county], self.election_precinct_map[county], county)
            self.matching[county] = solved_matching
        print(self.effective_dummies, len(self.ga_precinct_map))
    
    def match_precincts_in_county(self, ga_county_precincts, election_county_precincts, county_name):
        """
        ga_county_precincts: list of precincts
        election_county_precincts: list of precincts
        """
        assert len(ga_county_precincts) <= len(election_county_precincts)
        num_dummies = len(election_county_precincts) - len(ga_county_precincts)
        print(num_dummies)
        if num_dummies > 0:
            self.effective_dummies += num_dummies - 1
        dummy_players = [Player('DUMMY_{}'.format(i)) for i in range(num_dummies)]
        ga_county_players = {precinct: Player(precinct) for precinct in ga_county_precincts}
        self.ga_player_map[county_name] = ga_county_players
        election_county_players = {precinct: Player(precinct) for precinct in election_county_precincts}

        for player in ga_county_players.values():
            pref_list = self.preference_list(player, election_county_precincts, election_county_players)
            player.set_prefs(pref_list)

        election_county_player_list = list(election_county_players.values())
        for player in dummy_players:
            player.set_prefs(election_county_player_list)
        
        for player in election_county_players.values():
            pref_list = self.preference_list(player, ga_county_precincts, ga_county_players)
            player.set_prefs(pref_list + dummy_players)
        
        suitors = list(ga_county_players.values()) + dummy_players
        # print(suitors)
        reviewers = list(election_county_players.values())
        game = StableMarriage(suitors, reviewers)
        solved_game = game.solve()
        # print(solved_game)
        return solved_game

    def preference_list(self, player, match_precinct_list, match_player_dict):
        # list of tuples of form (precinct, score)
        score_list = process.extract(player.name, match_precinct_list, limit=None)
        pref_list = [match_player_dict[tup[0]] for tup in score_list]
        return pref_list

    def get_match(self, precinct, county):
        """
        returns a mapping from the given precinct shapefile mapping to election
        data precinct and county tuple
        """
        # print(county, type(self.matching[county]))
        player_from_precinct = self.ga_player_map[county][precinct]
        return self.matching[county][player_from_precinct].name

    def dump(self, filename):
        """
        dumps the matchings as a json
        """
        # d is of the format {county: {shape_precinct: election_precinct}}
        d = defaultdict(dict)
        for county, ga_county_players in self.ga_player_map.items():
            for shape_precinct, ga_player in ga_county_players.items():
                election_precinct = self.matching[county][ga_player].name
                d[county][shape_precinct] = election_precinct

        with open(filename, 'w') as f:
            json.dump(d, f)
