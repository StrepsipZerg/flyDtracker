# FlyDtracker
classes & functions currently defined in the working_functions .py file :
  -fly_data class ; is called by get_ori to order info output
  -relative_fly class ; is used to compute relative information between 2 flies
  -get_ori function ; gets raw orientations and directions from numpy array of positions and video file.
  -get_angle function ; Outputs direction, speed and distance for a given frame based on the positions of one fly
  -convert180to360 function ; converts a set of angle from a 180degrees referencial to a 360degrees referencial
  -flip function ; adds a certain amount to a set of angles while keeping a 360degrees referencial
  -direc_stats function ; outputs information about a chunk of data. Should not be used as is.
  -angleFromPoints function ; calculate the angle corresponding to a vector of coordinates 0,0 -> x,y. 360degrees referencial
  -save_ori_pickle function ; get_ori for a certain session, and saves as pickle.
  -get_ori_pickle function ; load pickle for a certain session
