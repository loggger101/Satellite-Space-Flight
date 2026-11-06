function s = copyOptions(s, opts, map)
%COPYOPTIONS Copy the name-value options that were given into JSON keys.
%   s = copyOptions(s, opts, map) sets s.(map(k,2)) = opts.(map(k,1)) for
%   every option the caller supplied. Options left out are not written, so
%   the simulator's own default applies (it is the one place defaults live).
for k = 1:size(map, 1)
    if isfield(opts, map(k, 1))
        value = opts.(map(k, 1));
        if isstring(value) && isscalar(value)
            value = char(value);
        end
        s.(map(k, 2)) = value;
    end
end
end
