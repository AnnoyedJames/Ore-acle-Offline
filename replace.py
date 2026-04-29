import sys

with open('frontend/src/components/MessageBubble.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

start_idx = text.find('if (is3x3) {\n                          return (\n                            <div')
if start_idx != -1:
    end_idx = text.find('                        }\n                      }\n                    }\n                  }\n                }\n                \n                return (')
    if end_idx != -1:
        new_text = '''if (is3x3) {
                          return (
                            <CraftingGrid 
                              s0={gridCols[0][0]} s1={gridCols[0][1]} s2={gridCols[0][2]}
                              s3={gridCols[1][0]} s4={gridCols[1][1]} s5={gridCols[1][2]}
                              s6={gridCols[2][0]} s7={gridCols[2][1]} s8={gridCols[2][2]}
                              result="Unknown"
                            />
                          );
'''
        text = text[:start_idx] + new_text + text[end_idx:]
        with open('frontend/src/components/MessageBubble.tsx', 'w', encoding='utf-8') as f:
            f.write(text)
        print('SUCCESS')
    else:
        print('END NOT FOUND')
else:
    print('START NOT FOUND')