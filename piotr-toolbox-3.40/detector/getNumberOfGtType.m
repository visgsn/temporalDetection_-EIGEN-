function [typeCount] = getNumberOfGtType(gt, type)
%getNumberOfGtType is used to get the total number of ground truth
%elements, matching 'type', in the given ground truth cell array.
%
% INPUTS
%   gt        - Cell array of ground truths
%   type      - See description of bbGt -> evalRes:
%               gt0 -> 0 = count unignored GTs, 1 = count ignored GTs
%               gt  -> -1 = ignore,  0 = fn [unmatched],  1 = tp [matched]
%
% OUTPUT
%   typeCount - Number of detected items of type 'type'

%% Initialize variables
typeCount = 0;

%% Count occurrances of given type
for imgNr = 1:length(gt)
    numberOfGts = size(gt{imgNr}, 1);
    if numberOfGts ~= 0
        for gtNr = 1:numberOfGts
            if gt{imgNr}(gtNr, 5) == type
                typeCount = typeCount + 1;
            end
        end
    end
end


end

