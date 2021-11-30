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

    maze_args = {
        "pit_locs": [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
                      (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
                      (2, 5), (2, 6), (4, 6), (4, 7), (6, 6), (7, 6)],
        "fire_locs": [],
        "width": 7,
        "height": 7,
        "treasure_locs": [(7, 7)]
    }

    @staticmethod
    def source_maze() -> SvetlikGridWorldMDP:
        """Must pass next to fire to win"""
        return SvetlikGridWorldMDP(**SvetlikGridWorldEnvironments.maze_args,
                                   init_loc=(1, 3))

    @staticmethod
    def target_maze() -> SvetlikGridWorldMDP:
        """Must pass next to fire to win"""
        return SvetlikGridWorldMDP(**SvetlikGridWorldEnvironments.maze_args)

    args_1010 = {
        "pit_locs": [(2, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2),
                    (2, 4), (2, 5), (6, 5), (7, 5), (7, 7), (2, 8),
                    (3, 9), (8, 5), (8, 7), (8, 9), (9, 5), (10, 5),
                    (9, 7), (9, 9), (10, 9)],
        "fire_locs": [(2, 3), (3, 3), (3, 6), (5, 8)],
        "width": 10,
        "height": 10,
        "default_reward": -5,
        "treasure_locs": [(10, 10)]
    }

    @staticmethod
    def source_1010() -> SvetlikGridWorldMDP:
        """All obstacles removed except row 3"""
        return SvetlikGridWorldMDP(**SvetlikGridWorldEnvironments.args_1010, init_loc=(5,5))

    @staticmethod
    def target_1010() -> SvetlikGridWorldMDP:
        """Must pass next to fire to win"""
        return SvetlikGridWorldMDP(**SvetlikGridWorldEnvironments.args_1010)
