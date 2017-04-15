#root = ARGV[0].gsub("\\", "/")
root = "svg"

out_cpp = <<END
#include "pch.h"
#include "SvgIcons.h"

namespace imqs {
namespace anno {
namespace svg {

END

out_h = <<END
#pragma once

namespace imqs {
namespace anno {
namespace svg {

END

list = <<END
const char* All[] = {
END

Dir.glob("#{root}/*.svg") { |f|
	name = File.basename(f)[0..-5]
	nice_name = name
	var_name = name.gsub("-", "_")
	print name + "\n"
	xml = File.read(f)
	xml.gsub!('xmlns="http://www.w3.org/2000/svg" ', '')
	out_h += "extern const char* %s;\n" % [var_name]
	out_cpp += "const char* #{var_name} = R\"-("
	out_cpp += xml
	out_cpp += ")-\";\n\n"
	list += "\t\"#{nice_name}\", #{var_name},\n"
}

list += <<END
	nullptr, nullptr
};
END

out_cpp += list
out_cpp += <<END

void LoadAll(xo::Doc* doc) {
	for (size_t i = 0; All[i]; i += 2)
		doc->SetSvg(All[i], All[i + 1]);
}

}
}
}
END

out_h += <<END
void LoadAll(xo::Doc* doc);
}
}
}
END

File.write("src/SvgIcons.cpp", out_cpp)
File.write("src/SvgIcons.h", out_h)