# Defining functions, using conditions and loops

function f(x, y)                 # define a function
    if x < y                     # branch
        return sin(x + y)
    else
        return cos(x + y)
    end
end


function print_plurals(list_of_words)   # define a function
    for word in list_of_words           # loop
        println(word * "s")
    end
end

