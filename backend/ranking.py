class AlgorithmRanker:
    @staticmethod
    def get_best_algorithm(leaderboard):
        if not leaderboard:
            return None, "No algorithms to evaluate."
        best = max(leaderboard, key=lambda x: x.get("sei", 0))
        explanation = f"🏆 **{best['name']}** is recommended (SEI = {best['sei']}%)"
        return best["name"], explanation