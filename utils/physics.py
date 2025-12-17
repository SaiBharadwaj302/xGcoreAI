# utils/physics.py
import numpy as np

def calculate_visible_angle(x, y):
    """Calculates the visible angle of the goal from a pitch coordinate."""
    if x >= 120: x = 119.9
    a1 = np.arctan2(36 - y, 120 - x)
    a2 = np.arctan2(44 - y, 120 - x)
    return abs(a1 - a2)

def simulate_trajectory(start_x, start_y, velocity_ms, loft_deg, curl=0.0):
    """
    Simulates the 3D trajectory of the ball using basic physics (drag, gravity, Magnus effect).
    Returns status, end_height, flight_time, and trajectory coordinates.
    """
    g, m, rho, Cd, A, dt = 9.81, 0.45, 1.225, 0.25, 0.038, 0.01
    dist_yards = np.sqrt((120 - start_x)**2 + (40 - start_y)**2)
    target_dist_m = dist_yards * 0.9144 
    angle_rad = np.radians(loft_deg)
    vx = velocity_ms * np.cos(angle_rad)
    vy = velocity_ms * np.sin(angle_rad)
    lat_v, lat_pos_m = 0.0, 0.0
    x, y, t = 0, 0, 0
    path_x, path_y, path_z = [start_x], [start_y], [0]

    while x < target_dist_m + 2 and y >= -0.5:
        v = max(0.1, np.sqrt(vx**2 + vy**2))
        drag_force = 0.5 * rho * v**2 * Cd * A
        ax = -(drag_force * (vx/v)) / m
        ay = -g - (drag_force * (vy/v)) / m
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        # Magnus effect (curl/spin): linear lift coefficient with sane caps for stability
        # Amplify spin effect: map slider to lift coefficient with higher cap
        cl = np.clip(0.05 * curl, -0.35, 0.35)  # curl slider -> lift coefficient
        a_lat = (0.5 * rho * abs(cl) * A * v**2) / m
        a_lat *= np.sign(curl)  # direction of bend
        lat_v += a_lat * dt
        lat_v = np.clip(lat_v, -25.0, 25.0)
        lat_pos_m += lat_v * dt
        
        if target_dist_m > 0:
            ratio = x / target_dist_m
            base_px = start_x + (120 - start_x) * ratio
            base_py = start_y + (40 - start_y) * ratio
            final_py = base_py + (lat_pos_m / 0.9144)  # convert lateral meters back to yards
            path_x.append(base_px)
            path_y.append(final_py)
            path_z.append(max(0, y))

    final_h = 0.0
    status = "MISS_SHORT"
    for i in range(len(path_x)):
        if path_x[i] >= 120:
            final_h = path_z[i]
            final_w = path_y[i]
            if 36 <= final_w <= 44 and 0 < final_h < 2.44: status = "GOAL"
            elif final_h >= 2.44 and 36 <= final_w <= 44: status = "MISS_HIGH"
            elif final_w < 36 or final_w > 44: status = "MISS_WIDE"
            else: status = "MISS_SHORT"
            break
            
    if path_x[-1] < 120: status = "MISS_SHORT"
    return status, final_h, t, {"x": path_x, "y": path_y, "z": path_z}