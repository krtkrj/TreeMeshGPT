from fns import process_mesh, sample_point_cloud, quantize_remove_duplicates, prepare_halfedge_mesh, create_io_sequence

QUANT_BIT = 7
MESH_FN = "demo/objaverse_nut.obj"
N_TRIAL = 20
MAX_N_FACES = 5500
AUGMENTATION = True
    
def manifold_trial(mesh_fn, quant_bit):
    vertices, triangles = process_mesh(mesh_fn, augment = AUGMENTATION) # Open, augment, normalize mesh
    pc = sample_point_cloud(vertices, triangles, sampling='uniform') # Sample point cloud for training.
    vertices, triangles = quantize_remove_duplicates(vertices, triangles, quant_bit = quant_bit) # Quantize and remove duplicates
    o3d_half_edge_mesh, vertices, n_faces = prepare_halfedge_mesh(vertices, triangles)
    return o3d_half_edge_mesh, vertices, n_faces, pc

success = False 
for trial in range(N_TRIAL):  
    # 7-bit quantization with high face count tends to violate manifold condition. Keep trying by using different augmentation if fails.
    try:
        o3d_half_edge_mesh, vertices, n_faces, pc = manifold_trial(MESH_FN, QUANT_BIT)
        success = True  
        break  
    except Exception as e:
        continue
    
if success and n_faces <= MAX_N_FACES:
    edges, gt = create_io_sequence(o3d_half_edge_mesh)
                
    output_seq_no_aux_token = [elem for sublist in gt for elem in sublist]
    input_seq_no_aux_token = [elem for sublist in edges for elem in sublist]
    
    assert len(output_seq_no_aux_token) == len(input_seq_no_aux_token)
    n_seq = len(output_seq_no_aux_token)
    
    io_dict = {
        'vertices': vertices,
        'edges': edges,
        'n_faces': n_faces,
        'n_seq': n_seq,
        'pc': pc
    }
    
    print("IO creation successful. Mesh face count: {}. Sequence length: {}.".format(n_faces, n_seq))
else:
    print(f"Skipping sample.")