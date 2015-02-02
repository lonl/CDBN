function model= dbnFit(X, numhid, y, varargin)
%fit a DBN to bianry data in X

%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... list of numbers of hidden units
%y              ... List of discrete labels

%OUTPUTS:
%model          ... A cell array containing models from all RBM's

%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400

numopts=length(varargin);
H=length(numhid);
model=cell(H,1);
if H>=2
    
    %train the first RBM on data
    if(numopts>=1)
        model{1}= rbmBB(X, numhid(1),varargin{1});
    else
        model{1}= rbmBB(X, numhid(1));
    end
    
    %train all other RBM's on top of each other
    for i=2:H-1
        if(numopts>=i)
            model{i}=rbmBB(model{i-1}.top, numhid(i), varargin{i});
        else
            model{i}=rbmBB(model{i-1}.top, numhid(i));
        end
        
    end
    
    %the last RBM has access to labels too
    if(numopts>=H)
        model{H}= rbmFit(model{H-1}.top, numhid(end), y, varargin{H});
    else
        model{H}= rbmFit(model{H-1}.top, numhid(end), y);
    end
else
    
    %numhid is only a single layer... but we should work anyway
    if (numopts>=1)
        model{1}= rbmFit(X, numhid(1), y, varargin{1});
    else
        model{1}= rbmFit(X, numhid(1), y);
    end
end    

