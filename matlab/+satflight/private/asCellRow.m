function c = asCellRow(x, where)
%ASCELLROW A JSON list of objects as a 1-by-N cell array of structs.
%   jsondecode turns a list of objects into a struct array (or a single
%   struct when the list has one entry, or [] when it is empty); jsonencode
%   would write a 1x1 struct back as an object, not a list. A cell row always
%   encodes as a JSON list.
if iscell(x)
    c = reshape(x, 1, []);
elseif isstruct(x)
    c = reshape(num2cell(x), 1, []);
elseif isnumeric(x) && isempty(x)
    c = {};
else
    error("satflight:badScenario", "%s: expected a list of objects, got a %s.", where, class(x));
end
for k = 1:numel(c)
    if ~isstruct(c{k}) || ~isscalar(c{k})
        error("satflight:badScenario", "%s entry %d: expected an object, got a %s.", ...
            where, k, class(c{k}));
    end
end
end
