"""Contact proxy reward shared with the differentiable world model."""
from morphmimic.env.observations import _robot_state
from morphmimic.model.rewards_torch import contact_reward_terms


def contact_tracking(env, command_name, reward_cfg):
    command = env.command_manager.get_term(command_name)
    library = command.library
    contact = {}
    if hasattr(library, 'reference_floor'):
        clip = library.frame_clip[command.global_frame]
        contact = dict(reference_floor=library.reference_floor[clip],
                       reference_floor_valid=library.reference_floor_valid[clip])
    terms = contact_reward_terms(_robot_state(env, command), command.ref_state(), reward_cfg, **contact)
    # RewardManager applies dt; analytic terms already include that integration.
    return sum(terms.values()) / env.step_dt
