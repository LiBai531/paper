如何在source insight中使用astyle的代码整理功能？ 

用source insight 编辑代码时，苦于source insight没有集成的代码格式化工具， 
GNU的astyle是一个免费的代码格式化工具,能够整理符合c/c++规范 。      
我们可以将astyle.exe外挂到SourceInsight中。详细步骤如下： 

1：从 http://astyle.sourceforge.net 上下载AStyle_1.24_windows.zip(开放源码，可以仔细阅读)； 
2：加压缩后将bin文件夹下的astyle.exe放到C:\Program Files\astyle下 （自己可以放在任意位置） 
      在SourceInsight菜单栏里，Options-->Custom Commands界面上选择：Add,在弹出对话框写入 Astyle, 
3：在run中添加"C:\Program Files\astyle\astyle.exe" --style=ansi %f，
其中，如果astyle.exe所在路径中有空格，必须用＂＂括起来，参数--style=ansi 代表ansi C 格式
(如果你需要格式化java代码，这个地方改为：--style=java),
"%f"是指作用于当前文件，这个必须的．其它命令参数可以参考astyle的帮助参数 
可以查看网页http://astyle.sourceforge.net/astyle.html 
4：此外，在此界面上还可以为这个命令设置快捷键，点击＂keys＂，添加你觉得方便的按钮； 
5：在SourceInsight菜单栏里，Options-->Menu Assignments界面上，
将这个命令名称为Astyle添加到某工具栏下，我是依然放在了Option下，
在左面的Command列表里找到我们刚才添加的＂Astyle＂，在右面的Menu中选择你要加到那个菜单下，
这里我加到＂Option＂下，可以在＂Menu Contents＂选择适当位置，点击＂insert＂即可; 

C:\astyle\bin\Astyle.exe --style=ansi -s8 -S -N -L -m0 -M40 --convert-tabs --suffix=.pre %f

来自: http://hi.baidu.com/tihu1111/blog/item/91f77401c9e47a027aec2c63.html 













http://hi.baidu.com/jin_haiqing/blog/item/a3ee7cadb4f08cc27dd92a73.html

使用astyle格式化代码2010-09-04 21:29astyle是一款代码格式化工具，它的下载地址是：http://sourceforge.net/projects/astyle

一。基本命令

astyle --style=ansi main.cs (使用ansi风格格式化main.cs)

了解上面的命令就可以格式化一个文件了，下面来看如何格式化目录下的文件

二。格式化目录

for /R %f in (*.cpp;*.cs;) do astyle --style=ansi "%f" (使用ansi风格格式下当前目录下的所有cpp,cs文件,注意：批处理文件时，"%f" 要改为"%%f")
for /R %f in (*.c;*.h;) do "D:\Program Files\astyle\bin\astyle" --style=ansi "%f"
三。参数说明：

(1) -f
在两行不相关的代码之间插入空行，如import和public class之间、public class和成员之间等；
(2) -p
在操作符两边插入空格，如=、+、-等。
如：int a=10*60;
处理后变成int a = 10 * 60;
(3) -P
在括号两边插入空格。另，-d只在括号外面插入空格，-D只在里面插入。
如：System.out.println(1);
处理后变成System.out.println( 1 );
(4) -U
移除括号两边不必要的空格。
如：System.out.println( 1 );
处理后变成System.out.println(1);
(5) -V
将Tab替换为空格。

(6)-N

本条主要针对namespaces，如果没有此参数，效果如下：

namespace foospace
{
class Foo
{
    public:
        Foo();
        virtual ~Foo();
};
}
有此参数就会变成这样：

namespace foospace
{
    class Foo
    {
        public:
            Foo();
            virtual ~Foo();
    };
}

(7) -n

不生成备份文件，即默认的 .orig文件。

C#的默认方式为第二种，所以如果你是用来格式化C#代码的话，这个参数就有用了。

四：加入到VS2008,VS2005中

估计加入到VS2005中也是一样，不过我这里没有VS2005,就说一下VS2008的做法。

工具——>外部工具——>添加

标题：astyle 

命令：AStyle.exe （填好astyle.exe的路径）

参数：--style=allman -N $(ItemDir)$(ItemFileName)$(ItemExt)

初始目录：$(TargetDir)

勾上“使用初始目录”

点击确定完成。以后就可以在工具菜单中找到“astyle“这一项了，点击它，就可以对当前文件进行格式化操作。


五：加入到VS6中

Tools——>Customize——>Tools

标题：astyle 

命令：AStyle.exe （填好astyle.exe的路径）

参数：--style=ansi -s4 --suffix=.orig $(FileName)$(FileExt)

初始目录：$(FileDir)

勾上“Using Output Window”

点击确定完成。以后就可以在工具菜单中找到“astyle“这一项了，点击它，就可以对当前文件进行格式化操作。


六：加入到Ultraedit和UltraStudio


高级-->工具配置——>外部工具——>添加

命令：AStyle.exe -v --style=ansi -s4 --suffix=.orig "%f"（填好astyle.exe的路径）


Optiones：选择 Windows program和Save Active File.

Output: 选择output to list box,show dos box 和no replace。


点击确定完成。以后就可以在工具菜单中找到“astyle“这一项了，点击它，就可以对当前文件进行格式化操作。


七：加入到Source insight


Options-->Custom Command-->Add

Command：astyle

Run "astyle.exe" --style=ansi --pad=oper --unpad=paren -s4 --suffix=.orig %f（填好astyle.exe的路径）


Output：不选.

Control: 选择pause when done和exit to window.

source links in output:file, then line

-->menu

add to work menu.


点击确定完成。以后就可以在Work菜单中找到“astyle“这一项了，点击它，就可以对当前文件进行格式化操作。

八：控制台目录批处理(Astyle.bat)


REM bygreencn@gmail.com

REM 批量将本目录中的所有C++文件用Astyle进行代码美化操作

REM 2009-01-05

REM 设置Astyle命令位置和参数

@echo off

set astyle="astyle.exe"

REM 循环遍历目录

for /r . %%a in (*.cpp;*.c) do %astyle% --style=ansi --pad=oper --unpad=paren -s4 -n "%%a"

for /r . %%a in (*.hpp;*.h) do %astyle% --style=ansi --pad=oper --unpad=paren -s4 -n "%%a"

REM 删除所有的astyle生成文件

for /r . %%a in (*.orig) do del "%%a"

pause
 




