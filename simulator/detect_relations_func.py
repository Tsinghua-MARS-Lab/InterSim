for frame_idx1 in range(total_frame_number):
    if frame_idx1 < start_from_frame:
        continue
    if cross:
        break
    agent1 = get_agent_from_dic(agent_in_dic1, frame_idx1, agent_id1)
    if agent1.x == -1 or agent1.y == -1:
        continue
    for frame_idx2 in range(total_frame_number):
        if frame_idx2 < start_from_frame:
            continue
        if cross:
            break
        agent2 = get_agent_from_dic(agent_in_dic2, frame_idx2, agent_id2)
        if agent2.x == -1 or agent2.y == -1:
            continue
        cross1 = check_collision_for_two_agents_rotate_and_dist_check(checking_agent=agent1,
                                                                      target_agent=agent2)
        cross2 = check_collision_for_two_agents_rotate_and_dist_check(checking_agent=agent2,
                                                                      target_agent=agent1)
        cross = cross1 | cross2
        if cross:
            # print("cross detected: ", agent_id1, agent_id2, frame_idx1, frame_idx2)
            if abs(frame_idx1 - frame_idx2) < last_detection_frame_diff:
                last_detection_frame_diff = abs(frame_idx1 - frame_idx2)
                frame_diff = frame_idx1 - frame_idx2
                if frame_diff > 0:
                    edge_to_add = [agent_id2, agent_id1, frame_idx1, abs(frame_diff)]
                elif frame_diff < 0:
                    edge_to_add = [agent_id1, agent_id2, frame_idx2, abs(frame_diff)]
                else:
                    print("collide at same frame", agent_id1, agent_id2)
                    cross = False  # most likely a false detection