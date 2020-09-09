function [v, header] = readrec_V4_2(filename)
%
%  This will read in a rec file from the Philips scanner.
%  It will read in REC files even with real and imaginary data
%  as well as do the appropriate signal intensity adjustments
%  as the REC file stores uint16, but the underlying data can be 
%  positive or negative.
%

% Craig Jones (craig@mri.jhu.edu)  April 28, 2004
% 20040510 CJ - changed reading loop to be over size(A,1)
% 20040527 CJ - checked for 'rec' and 'REC'  
% 20040624 CJ - fixed TeX interpreter, 
% 20040708 CJ - correctly reads in types for 1.5T data
% 20040711 CJ - fixed number of dynamics
% 20040916 CJ - added in reading of echoes
% 20050503 CJ - fixed for reading version 3/4 PAR files.
% 20050520 CJ - check for existence of par file.
% 20050606 CJ - added for cardiac phases
% 20060303 CJ - Fixed problem with V in patient name
% 20060308 CJ - Fixed Jonathan's problems with passing in a par file

% Mina Kim (mina@mri.jhu.edu)    March 26,  2007
% 20070326 MK, JSG - Fixed for reading version V4.1 par file

% Alan Huang (Alan.J.Huang@gmail.com)   November, 2007
% 20071105 AH, JH - Fixed for reading version V4.2 par file
% 20071106 AH, JF, IL - Fixed for reading ASL images

%%=================================================================
%%
%%  Load in the PAR file and do the necessary conversions.
%%
%%=================================================================
if( strcmp(filename(end-2:end), 'par' ) )
    filename_par = filename;
    filename = strrep(filename, 'par', 'rec');
elseif( strcmp(filename(end-2:end), 'PAR' ) )
    filename_par = filename;
    filename = strrep(filename, 'PAR', 'REC');
elseif( strcmp(filename(end-2:end), 'rec') )
	filename_par = strrep(filename, 'rec', 'par');
elseif( strcmp(filename(end-2:end), 'REC') )
	filename_par = strrep(filename, 'REC', 'PAR');
else
    error(sprintf('Yo dude, %s does not seem to be a PAR or REC file', filename));
end

[nrows, ncols, nslices, nechoes, ndynamics, nphases, A, header, par_version, nASL_type] = ...
    parseHeader(filename_par);
% [nrows, ncols, nslices, nechoes, ndynamics, nphases, A, header, par_version, nASL_type] = ...
%     mjd_parseheader(filename_par);

if( nrows == 0 ) 
	return;
end

%%=================================================================
%%
%%  Number of types of scans:  0=magnitude, 1 = Real, 2 = Imaginary
%%
%%=================================================================
if( par_version == 4 || par_version == 4.1 || par_version == 4.2)
	typeind = 6;
elseif( par_version == 3 )
	typeind = 5;
end
types = unique(A(:,typeind));
dynamics = unique(A(:,3));
echoes = unique(A(:,2));

if( par_version == 4.2)
    ASL_type = unique(A(:,49));
    v = zeros([nrows ncols nslices nechoes ndynamics length(types) nphases nASL_type]);
else
    v = zeros([nrows ncols nslices nechoes ndynamics length(types) nphases]);
end

fprintf('Reading in %s:\n', filename);
fprintf('\tnrows = %d\n', nrows);
fprintf('\tncols = %d\n', ncols);
fprintf('\tnslices = %d\n', nslices);
fprintf('\tndynamics = %d\n', ndynamics);
fprintf('\tnechoes = %d\n', nechoes);
fprintf('\ttypes = %d\n', length(types));
fprintf('\tphases = %d\n', nphases);
if( par_version == 4.2)
    fprintf('\tnASL_type = %d\n', nASL_type);
end

%%=================================================================
%%  
%%  Read in the data.
%% 
%%=================================================================
if( findstr(filename, '.gz') )
	tmpname = tempname;
	unix(sprintf('gzcat %s > %s', filename, tmpname));
else 
	tmpname = filename;
end

fp = fopen(tmpname, 'rb', 'l');

if( fp == -1 )
	if( isunix )
		person = getenv('USER');
	elseif( ispc )
		person = getenv('USERNAME');
	else
		person = 'matlab user';
	end

	error(sprintf('readrec:  I''m sorry, %s, the file %s does not exist', getenv('USER'), filename_par));
end

h = waitbar(0,''); set(findall(h,'type','text'),'Interpreter','none', 'string', sprintf('Reading in %s', filename));
for ii=1:size(A,1)

	%  Determeine the rescale slope, intercept and other scaling factors.
	if( par_version == 3 )
		rs = A(ii,9);  ri = A(ii,8); ss = A(ii,10);
	elseif( par_version == 4 || par_version == 4.1 || par_version == 4.2)   
		rs = A(ii,13);  ri = A(ii,12); ss = A(ii,14);
    end

    if (par_version == 4.2)
        v(:,:,A(ii,1),find(echoes==A(ii,2)),find(dynamics==A(ii,3)),find(types==A(ii,typeind)),A(ii,4), find(ASL_type==A(ii,49))) = ...
			(fread(fp, [nrows, ncols], 'int16') * rs + ri) / (rs * ss);
        waitbar(ii/length(A),h)
    else 
        v(:,:,A(ii,1),find(echoes==A(ii,2)),find(dynamics==A(ii,3)),find(types==A(ii,typeind)),A(ii,4)) = ...
			(fread(fp, [nrows, ncols], 'int16') * rs + ri) / (rs * ss);
        waitbar(ii/length(A),h)
    end
end
close(h);

fclose(fp);

v = permute(v, [2 1 3:length(size(v))]);

if( findstr(filename, '.gz') )
	delete(tmpname);
end

%========================================================================
function [nrows, ncols, nslices, nechoes, ndynamics, nphases, A, header, par_version, nASL_type] = parseHeader(filename_par)
%  parseHeader - Parse a Philips PAR file for some important parameters
%

%  Craig Jones (craig@mri.jhu.edu)  
%  20040616 - fixed up reading in the header for stupid Windoze boxes

%  Alan Huang (Alan.J.Huang@gmail.com)
%  20071106 - added nASL_type parameter for patch 2.5 (V4.2)

nrows = 0; ncols = 0; nslices = 0; nechoes = 0; ndynamics = 0; A = []; header = []; par_version = 0;

header.filename = filename_par;

line = '';
fp = fopen(filename_par, 'rt');

if( fp == -1 )
	if( isunix )
		person = getenv('USER');
	elseif( ispc )
		person = getenv('USERNAME');
	else
		person = 'matlab user';
	end

	disp(sprintf('readrec:  I''m sorry, %s, the file %s does not exist', getenv('USER'), filename_par));
	return;
end

firstheader = 1;
par_version = 0;

%line = fgetl(fp);
while( 1 )
    line = fgetl(fp);

    if (( strncmp('#sl', line, 3) == 1) || ( strncmp('# sl', line, 4) == 1) || ( strncmp('#  sl ec', line, 8) == 1)), break, end; 
    
	if( firstheader & line(1) == '#' ) 
		[s,f] = regexp(line, '\w*[vV][0-9](\.[0-9]){0,1}\w*'); 
		if( ~isempty(s) )
			par_version = str2num( line(s(end)+1:f(end)) );
		end
	end

	%% We are not the in first header any more
    if ( strncmp('#', line, 1) ~= 1) 
		firstheader = 0;
	end

    %%  Look for off center values
    if( findstr('Off Centre ', line) > 0 )
        aa = line(findstr(':', line)+1:end);
       	header.off_center = str2num(aa);
		header.annotation = 'ap fh rl';
    end

    %%  Look for off center angulation
    if( findstr('Angulation ', line) > 0 )
        aa = line(findstr(':', line)+1:end);
       	header.angulation = str2num(aa);
    end

    %%  Look for the FOV
    if( findstr('FOV (', line) > 0 )
        aa = line(findstr(':', line)+1:end);
       	header.fov = str2num(aa);
    end

    %%  Look for number of rows and columns
    if( findstr('Recon resolution', line) > 0 )
        aa = line(findstr(':', line)+1:end);
        aa = str2num(aa);
        nrows = aa(1);  ncols = aa(2);
		header.recon_resolution = aa;
    end

    %%  Look for number of slices
    if( findstr('number of slices', line) > 0 )
        aa = line(findstr(':', line)+1:end);
        aa = str2num(aa);
        nslices = aa(1);
    end

    %%  Look for number of slices
    if( findstr('number of cardiac phases', line) > 0 )
        aa = line(findstr(':', line)+1:end);
        aa = str2num(aa);
        nphases = aa(1);
    end

%    %%  Look for Dataset name which will tell us the PAR file type
%	%%   ie whether acquired on the 1.5T or 3T
%    if( findstr('Dataset name:', line) > 0 )
%        tt = findstr(':', line);  %% There may be a : in the dir
%        aa = line(tt(1)+1:end);
%		
%		if( findstr('X:', aa) > 0 )
%			scanner=15;
%		elseif( findstr('intera', aa) > 0 )	
%			scanner=30;
%		else
%			error(sprintf('Could not find "Dataset name": in %s', filename_par));
%		end
%    end
end

line = fgetl(fp);


A = [];
ii = 1;
while( 1 )

    line = fgetl(fp);
    if( length(line) < 2 ), break; end
%    if( strncmp('# =', line, 3) == 1) break; end
    A(ii,:) = str2num(line);
    ii = ii + 1;
end

ndynamics = length(unique(A(:,3)));
nechoes = length(unique(A(:,2)));

if( par_version == 4.2)
    nASL_type = length(unique(A(:,49)));
else
    nASL_type = [];
end


%  Added for the new PAR files.
if( nrows == 0 )
	nrows = A(1,10);
	ncols = A(1,11);
end

fclose(fp);
