function parrec_to_nii(filename_in, filename_out)

    % wraps readrec_V4_2, make_nii and save_nii into a single function that
    % reads in the path to a PARREC file and outputs a NiFTI
    % input and output filenames are identical except
    % the extension is change from .PAR to .nii.gz
    % note that this function assumes a voxel size of [1 1 1] and so won't work
    % if the voxel dimensions differ from this

    [v, header] = readrec_V4_2(filename_in);
    nii = make_nii(v);
    save_nii(nii, filename_out);

    exit
