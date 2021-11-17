from svetlik_gridworld import SvetlikGridWorldMDP

class SvetlikGridWorldEnvironments:
    @staticmethod
    def source_55() -> SvetlikGridWorldMDP:
        """Empty (no obstacles)"""
        return SvetlikGridWorldMDP(
            pit_locs=[],
            fire_locs=[],
            width=5,
            height=5,
            treasure_locs=[(5, 5)])  # ~750 steps to converge

    @staticmethod
    def target_55() -> SvetlikGridWorldMDP:
        return SvetlikGridWorldMDP(
            pit_locs=[(2, 2), (4, 2)],
            fire_locs=[(2, 4), (3, 4)],
            width=5,
            height=5,
            treasure_locs=[(5, 5)])  # ~900 steps to converge

    @staticmethod
    def source_77() -> SvetlikGridWorldMDP:
        """All obstacles removed except row 3"""
        return SvetlikGridWorldMDP(
            pit_locs=[(3, 3)],
            fire_locs=[(1, 3), (2, 3), (5, 3), (6, 3), (7, 3)],
            width=7,
            height=7,
            treasure_locs=[(7, 7)])

    @staticmethod
    def target_77() -> SvetlikGridWorldMDP:
        """Must pass next to fire to win"""
        return SvetlikGridWorldMDP(
            pit_locs=[(3, 3), (4, 5), (6, 6), (7, 5)],
            fire_locs=[(1, 3), (1, 7), (2, 3), (5, 3), (6, 3), (7, 3)],
            width=7,
            height=7,
            treasure_locs=[(7, 7)])
